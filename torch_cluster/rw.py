import warnings
import math

import torch
import torch_cluster.rw_cpu

if torch.cuda.is_available():
    import torch_cluster.rw_cuda


def random_walk(row, col, start, walk_length, p=1, q=1, coalesced=False,
                num_nodes=None, num_walks_per_node=1, weights=None):
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)` as described in the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper.
    Edge indices :obj:`(row, col)` need to be coalesced/sorted according to
    :obj:`row` (use the :obj:`coalesced` attribute to force).

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        start (LongTensor): Nodes from where random walks start.
        walk_length (int): The walk length.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        coalesced (bool, optional): If set to :obj:`True`, will coalesce/sort
            the graph given by :obj:`(row, col)` according to :obj:`row`.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        num_walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        weights (FloatTensor, optional): weights tensor of edges (default: :obj:onesTensor)

    :rtype: :class:`LongTensor`
            :class:`Node2vecWalk` class when p!=1 and q!=1
    """
    assert row.size(0) == col.size(0)
    if num_nodes is None:
        num_nodes = max(row.max(), col.max()).item() + 1

    if coalesced:
        _, perm = torch.sort(row * num_nodes + col)
        row, col = row[perm], col[perm]

    start = start.flatten()

    if weights:
        assert row.size(0) == weights.size(0)
        weights = weights.to(torch.float64)
    else:
        weights = row.new_ones(row.size(0), dtype=torch.float64)

    if math.isclose(p, 1) and math.isclose(q, 1):
        if row.is_cuda:
            return torch_cluster.rw_cuda.rw(row, col, start, walk_length, p, q,
                                        num_nodes)
        else:
            return torch_cluster.rw_cpu.rw(row, col, start, walk_length, p, q,
                                        num_nodes)
    else:
        if row.is_cuda:
            if p != 1 or q != 1:  # pragma: no cover
                warnings.warn('Parameters `p` and `q` are not supported yet and will'
                    'be restored to their default values `p=1` and `q=1`.')
                p = q = 1
            return torch_cluster.rw_cuda.rw(row, col, start, walk_length, p, q,
                                           num_nodes)
        else:
            return torch_cluster.rw_cpu.Node2vecWalk(row, col, start, walk_length, p, q,
                                            num_nodes, num_walks_per_node, weights)
