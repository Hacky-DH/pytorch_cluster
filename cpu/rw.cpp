#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <unordered_map>

#include "utils.h"
#include "alias.h"

at::Tensor rw(at::Tensor row, at::Tensor col, at::Tensor start,
size_t walk_length, float p, float q, size_t num_nodes) {
    auto deg = degree(row, num_nodes);
    auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);

    auto rand = at::rand({start.size(0), (int64_t)walk_length},
    start.options().dtype(at::kFloat));
    auto out = at::full({start.size(0), (int64_t)walk_length + 1}, -1, start.options());

    auto deg_d = deg.data<int64_t>();
    auto cum_deg_d = cum_deg.data<int64_t>();
    auto col_d = col.data<int64_t>();
    auto start_d = start.data<int64_t>();
    auto rand_d = rand.data<float>();
    auto out_d = out.data<int64_t>();

    for (ptrdiff_t n = 0; n < start.size(0); n++) {
        int64_t cur = start_d[n];
        auto i = n * (walk_length + 1);
        out_d[i] = cur;

        for (ptrdiff_t l = 1; l <= (int64_t)walk_length; l++) {
            cur = col_d[cum_deg_d[cur] +
                int64_t(rand_d[n * walk_length + (l - 1)] * deg_d[cur])];
            out_d[i + l] = cur;
        }
    }

    return out;
}

using torch::Tensor;

class Node2vecWalk {
public:
    typedef int64_t num_t;

    /*
    :param src: tensor, src indices of nodes in graph, (num_edges, 1)
    :param dst: tensor, dst indices of nodes in graph, (num_edges, 1)
    :param start: tensor, start indices of nodes in graph
    :param walk_length: int64
    :param p: float, return param p
    :param q: float, in-out param q
    :param num_nodes: num nodes
    :param num_walks_per_node: int64, num walks per node
    :param weights: tensor, weights of edges, (num_edges, 1)
    */
    explicit Node2vecWalk(const Tensor& src, const Tensor& dst, const Tensor& start,
        size_t walk_length, float p, float q, size_t num_nodes,
        size_t num_walks_per_node, const Tensor& weights):
    src(src), dst(dst), start(start), weights(weights),
    num_nodes(num_nodes), num_walks_per_node(num_walks_per_node),
    walk_length(walk_length), p(p<0?1:p), q(q<0?1:q) {
        preprocess_transition_probs();
    }

    Tensor operator()() const {
        return simulate_walks();
    }

    Tensor simulate_walks() const;

private:
    typedef std::tuple<num_t, num_t> edge_t;
    struct EdgeHash : public std::unary_function<edge_t, std::size_t> {
        std::size_t operator() (const edge_t& k) const {
            std::size_t seed = std::hash<num_t>()(std::get<0>(k));
            return seed ^ std::hash<num_t>()(std::get<1>(k));
        }
    };

    struct EdgeEqual : public std::binary_function<edge_t, edge_t, bool> {
        bool operator() (const edge_t& l, const edge_t& r) const {
            return (std::get<0>(l) == std::get<0>(r) &&
                std::get<1>(l) == std::get<1>(r));
        }
    };
    typedef std::unordered_map<edge_t, AliasSample, EdgeHash, EdgeEqual> edgeMap;

private:
    Tensor src;
    Tensor dst;
    Tensor start;
    Tensor weights;

    num_t num_nodes;
    num_t num_walks_per_node;
    num_t walk_length;

    float p;
    float q;

    std::vector<std::tuple<Tensor, Tensor>> neighbors;  // (dst, weights)
    std::vector<AliasSample> alias_nodes;
    edgeMap alias_edges;

    void preprocess_transition_probs();
    void node2vec_walk(num_t start_node, Tensor& walk);
};

// Preprocessing of transition probabilities for guiding the random walks.
void Node2vecWalk::preprocess_transition_probs() {
    for (num_t i = 0; i < num_nodes; ++i) {
        auto src_index = (src == i).nonzero();
        auto nb_weights = weights.index(src_index);
        // save neighbor
        neighbors.push_back(std::make_tuple(dst.index(src_index), nb_weights));
        nb_weights = nb_weights / nb_weights.sum();  // normalize
        alias_nodes.push_back(AliasSample(nb_weights));
    }
    for (num_t edge_index = 0; edge_index < src.size(0); ++edge_index) {
        auto src_ = src[edge_index].item<num_t>();
        auto dst_ = dst[edge_index].item<num_t>();
        Tensor dst_nbrs, weights_;
        std::tie(dst_nbrs, weights_) = neighbors[dst_];
        auto edge_probs = torch::zeros(dst_nbrs.size(0), torch::kFloat32);
        for (num_t nbr_index = 0; nbr_index < dst_nbrs.size(0); ++nbr_index) {
            auto dst_nbr = dst_nbrs[nbr_index].item<num_t>();
            auto weight = weights_[nbr_index].item<float>();
            auto dtx = (std::get<0>(neighbors[dst_nbr]) == src_).nonzero();
            if (dst_nbr == src_)  // dtx = 0
                edge_probs[nbr_index] = weight / p;
            else if (dtx.size(0) > 0)  // dtx = 1
                edge_probs[nbr_index] = weight;
            else  // dtx = 2
                edge_probs[nbr_index] = weight / q;
        }
        edge_probs = edge_probs / edge_probs.sum();  // normalize
        alias_edges.insert({std::make_tuple(src_, dst_), AliasSample(edge_probs)});
    }
}

Tensor Node2vecWalk::simulate_walks() const {
    auto all_nodes = start.size(0) * num_walks_per_node;
    auto walks = torch::full({all_nodes, walk_length}, -1, torch::kInt64);
    num_t node, cur, prev;
    for (num_t iter = 0; iter < all_nodes; ++iter) {
        node = iter % start.size(0);
        auto walk = walks[iter];
        walk[0] = node;
        for (num_t w = 1; w < walk_length; ++w) {
            cur = walk[w-1].item<num_t>();
            auto cur_nbrs = std::get<0>(neighbors[cur]);
            if (cur_nbrs.size(0) <= 0) break;
            if (w <= 1) {
                walk[w] = cur_nbrs[alias_nodes[cur]()].item<num_t>();
            } else {
                prev = walk[w-2].item<num_t>();
                const auto edge_table = alias_edges.find(std::make_tuple(prev, cur));
                if (edge_table == alias_edges.end()) break;
                const auto& as = edge_table->second;
                walk[w] = cur_nbrs[as()].item<num_t>();
            }
        }
    }
    return walks;
}

using pybind11::class_;
using pybind11::init;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rw", &rw, "Random Walk Sampling (CPU)");
    class_<Node2vecWalk>(m, "Node2vecWalk",
                         "Random Walk Sampling for Node2Vec(CPU)")
        .def(init<const Tensor&, const Tensor&, const Tensor&,
             size_t, float, float, size_t, size_t, const Tensor&>())
        .def("__call__", &Node2vecWalk::operator());
}
