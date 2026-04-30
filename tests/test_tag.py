"""Tests for core/tag.py"""

import pytest

from agentic_capsules.core.tag import TagDimension, TagSpace


# ---------------------------------------------------------------------------
# TagDimension
# ---------------------------------------------------------------------------

def test_dimension_len():
    d = TagDimension("doc_id", list(range(10)))
    assert len(d) == 10


def test_dimension_empty_raises():
    with pytest.raises(ValueError):
        TagDimension("empty", [])


# ---------------------------------------------------------------------------
# TagSpace — single dimension
# ---------------------------------------------------------------------------

def test_enumerate_single_dimension():
    space = TagSpace("analyst", [TagDimension("doc_id", [1, 2, 3])])
    tags = space.enumerate()
    assert len(tags) == 3
    assert tags[0].agent_name == "analyst"
    assert "doc_id=1" in tags[0].task_id
    assert "doc_id=3" in tags[2].task_id


def test_size_single_dimension():
    space = TagSpace("analyst", [TagDimension("doc_id", list(range(100)))])
    assert space.size == 100


# ---------------------------------------------------------------------------
# TagSpace — cross-product (two dimensions)
# ---------------------------------------------------------------------------

def test_enumerate_cross_product():
    space = TagSpace(
        "analyst",
        [
            TagDimension("doc_id", [1, 2]),
            TagDimension("analysis_type", ["sentiment", "summary"]),
        ],
    )
    tags = space.enumerate()
    assert len(tags) == 4  # 2 × 2
    task_ids = {t.task_id for t in tags}
    assert "doc_id=1__analysis_type=sentiment" in task_ids
    assert "doc_id=2__analysis_type=summary" in task_ids


def test_size_cross_product():
    space = TagSpace(
        "analyst",
        [
            TagDimension("doc_id", list(range(10))),
            TagDimension("type", ["a", "b", "c"]),
        ],
    )
    assert space.size == 30


# ---------------------------------------------------------------------------
# TagSpace — partition
# ---------------------------------------------------------------------------

def test_partition_exact_batches():
    space = TagSpace("analyst", [TagDimension("doc_id", list(range(10)))])
    batches = space.partition(k=5)
    assert len(batches) == 2
    assert all(len(b) == 5 for b in batches)


def test_partition_remainder():
    space = TagSpace("analyst", [TagDimension("doc_id", list(range(11)))])
    batches = space.partition(k=5)
    assert len(batches) == 3
    assert len(batches[0]) == 5
    assert len(batches[2]) == 1


def test_partition_k1_is_fine_grained():
    space = TagSpace("analyst", [TagDimension("doc_id", list(range(5)))])
    batches = space.partition(k=1)
    assert len(batches) == 5
    assert all(len(b) == 1 for b in batches)


def test_partition_k_equals_n_is_single_batch():
    space = TagSpace("analyst", [TagDimension("doc_id", list(range(10)))])
    batches = space.partition(k=10)
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_partition_invalid_k_raises():
    space = TagSpace("analyst", [TagDimension("doc_id", [1])])
    with pytest.raises(ValueError):
        space.partition(k=0)


# ---------------------------------------------------------------------------
# TagSpace — no dimensions raises
# ---------------------------------------------------------------------------

def test_empty_dimensions_raises():
    with pytest.raises(ValueError):
        TagSpace("analyst", [])
