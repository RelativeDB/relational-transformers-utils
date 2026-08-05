from __future__ import annotations

import numpy as np
import pytest

from relational_transformers_utils.graph import ContextGraph, ContextTruncated
from relational_transformers_utils.walks import (
    StdRng,
    rand_sample,
    reference_walk_counts,
    stdrng_first_u64_batch,
)


def test_stdrng_streams_are_deterministic_and_seed_sensitive():
    a = StdRng(123456789)
    b = StdRng(123456789)
    assert [a.u32() for _ in range(64)] == [b.u32() for _ in range(64)]
    assert StdRng(1).u64() != StdRng(2).u64()
    assert stdrng_first_u64_batch([1, 2]).tolist() == [StdRng(1).u64(), StdRng(2).u64()]


def test_stdrng_range_respects_bounds():
    rng = StdRng(7)
    draws = [rng.range(10, 3) for _ in range(500)]
    assert min(draws) >= 3 and max(draws) <= 9
    assert len(set(draws)) == 7
    with pytest.raises(ValueError):
        rng.range(3, 3)


def test_rand_sample_is_a_valid_deterministic_selection():
    for length, amount in ((20, 5), (400, 15), (400, 200), (1000, 170)):
        first = rand_sample(StdRng(11), length, amount)
        second = rand_sample(StdRng(11), length, amount)
        assert first == second
        assert len(first) == amount
        assert len(set(first)) == amount
        assert all(0 <= i < length for i in first)
    with pytest.raises(ValueError):
        rand_sample(StdRng(1), 5, 6)


def test_walk_counts_visit_only_eligible_reachable_nodes():
    # 0 -> [1, 2], 1 -> [2], 2 -> [] ; node 3 is disconnected.
    offsets = [0, 2, 3, 3, 3]
    neighbors = [1, 2, 2]
    counts = reference_walk_counts(4, offsets, neighbors, target=0,
                                   eligible=[0, 1, 1, 1], seed=99,
                                   num_walks=64, walk_length=8)
    assert counts[0] == 0          # ineligible
    assert counts[3] == 0          # unreachable
    assert counts[1] > 0 and counts[2] > 0
    again = reference_walk_counts(4, offsets, neighbors, target=0,
                                  eligible=[0, 1, 1, 1], seed=99,
                                  num_walks=64, walk_length=8)
    assert counts.tolist() == again.tolist()
    empty = reference_walk_counts(4, offsets, neighbors, 0, [0, 1, 1, 1], 99, 0, 8)
    assert empty.sum() == 0


class _Policy:
    seed = 42
    local_context_cells = 100
    max_context_cells = 64
    bfs_width = 8
    num_walks = 32
    walk_length = 6
    prefer_latest = False


def _graph():
    # customer 0 with orders 1..3; a peer customer 4 with order 5.
    node_ts = [np.nan, 1.0, 2.0, 3.0, np.nan, 2.5]
    node_cells = [3, 2, 2, 2, 3, 2]
    node_table = [0, 1, 1, 1, 0, 1]
    node_is_task = [0, 0, 0, 0, 0, 0]
    edge_parent = [0, 0, 0, 4]
    edge_child = [1, 2, 3, 5]
    return ContextGraph(node_ts, node_cells, node_table, node_is_task,
                        edge_parent, edge_child)


def test_context_graph_assembles_target_neighbourhood_deterministically():
    graph = _graph()
    ordered, focal = graph.assemble(0, cutoff_ts=10.0, eligible=None,
                                    policy=_Policy())
    again, _ = graph.assemble(0, cutoff_ts=10.0, eligible=None,
                              policy=_Policy())
    assert ordered.tolist() == again.tolist()
    assert ordered[0] == 0 and focal[0] == 1
    assert set(ordered.tolist()) <= {0, 1, 2, 3}   # peer branch needs the walk

    with_walk, _ = graph.assemble(0, cutoff_ts=10.0,
                                  eligible=np.asarray([0, 0, 0, 0, 1, 0],
                                                      np.uint8),
                                  policy=_Policy())
    assert set(with_walk.tolist()) >= {0}


def test_context_graph_reports_truncation_instead_of_dropping():
    graph = _graph()
    with pytest.raises(ContextTruncated):
        graph.assemble(0, cutoff_ts=10.0, eligible=None, policy=_Policy(),
                       max_nodes=2)


class _WalkPolicy:
    seed = 9
    local_context_cells = 100
    max_context_cells = 100
    bfs_width = 4
    num_walks = 256
    walk_length = 6
    prefer_latest = True


def test_context_graph_walk_ranks_peers_and_fallback_pads():
    # target 0 shares parent 6 with peers 1..3; nodes 4-5 are fallback pool.
    node_ts = [5.0, 1.0, 2.0, 3.0, 1.5, 2.5, np.nan]
    node_cells = [1, 1, 1, 1, 1, 1, 1]
    node_table = [0, 0, 0, 0, 0, 0, 1]
    node_is_task = [1, 1, 1, 1, 1, 1, 0]
    edge_parent = [6, 6, 6, 6]
    edge_child = [0, 1, 2, 3]
    graph = ContextGraph(node_ts, node_cells, node_table, node_is_task,
                         edge_parent, edge_child)
    eligible = np.asarray([1, 1, 1, 1, 1, 1, 0], np.uint8)

    ordered, focal = graph.assemble(0, cutoff_ts=10.0, eligible=eligible,
                                    policy=_WalkPolicy(),
                                    fallback_base=4, fallback_n=2)
    emitted = set(ordered.tolist())
    assert 0 in emitted
    assert emitted & {1, 2, 3}, "walk-ranked peers must enter the context"
    assert emitted & {4, 5}, "the fallback stage must pad from its pool"
    assert focal[list(ordered.tolist()).index(0)] == 1

    again, _ = graph.assemble(0, cutoff_ts=10.0, eligible=eligible,
                              policy=_WalkPolicy(),
                              fallback_base=4, fallback_n=2)
    assert ordered.tolist() == again.tolist()

    # a cutoff below every timestamp admits no dated peers
    tight, _ = graph.assemble(6, cutoff_ts=0.5, eligible=eligible,
                              policy=_WalkPolicy())
    assert set(tight.tolist()) == {6}
