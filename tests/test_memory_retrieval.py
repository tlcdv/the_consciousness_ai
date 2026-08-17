"""
The workspace memory bid was pinned to its floor by a stub, not by the architecture.

`get_similar_experiences` queried `PineconeIndexStub`, whose `query` returns one match
with `score` hardcoded to 0.0 and whose `upsert` is `pass`. The workspace bid at
train_rlhf.py:1061 guards on `score > 0.0`, so it could never fire. Measured consequence:
`bid_memory` read exactly 0.100000000 at 24,000 of 24,000 steps across 3 seeds
(docs/results/workspace_bids_live_2026_08.md).

The experiences were never missing. `store_experience` appends every one of them to
`recent_experiences`, and that append sits OUTSIDE the `attention_level >= 0.7` gate, so
it happens on every step. Retrieval simply searched somewhere else.

These tests pin the repair and the two facts that make it correct:

  - the default path is unchanged, including the 0.0 score, so the baseline stays
    bit-identical
  - `recent_experiences` is populated regardless of `attention_level`, which is why
    retrieval over it is unaffected by the phi gate that measured 0 of 8000 steps passing

They do NOT assert that the workspace competition improves. It does not: vision bids
exactly 1.0 and the memory bid caps at 0.6, so the winner is unchanged by construction.
"""
import torch

from models.memory.memory_core import MemoryConfig, MemoryCore


def _core(enable_retrieval: bool) -> MemoryCore:
    return MemoryCore(MemoryConfig(enable_retrieval=enable_retrieval))


def _store(core: MemoryCore, state: torch.Tensor, attention: float = 0.9) -> None:
    core.store_experience(
        state=state,
        action=torch.zeros(4),
        reward=0.0,
        emotion_values={"valence": 0.0, "arousal": 0.0},
        attention_level=attention,
    )


# --- the default path must not move -------------------------------------------------

def test_default_is_off():
    assert MemoryConfig().enable_retrieval is False


def test_default_path_still_returns_the_zero_score_stub():
    """The bug is preserved when the flag is off, so the baseline is bit-identical."""
    core = _core(enable_retrieval=False)
    _store(core, torch.ones(16))
    results = core.get_similar_experiences(torch.ones(16), k=1)
    assert results[0]["score"] == 0.0


# --- the repaired path ---------------------------------------------------------------

def test_identical_query_scores_near_one():
    core = _core(enable_retrieval=True)
    _store(core, torch.ones(16))
    results = core.get_similar_experiences(torch.ones(16), k=1)
    assert results[0]["score"] > 0.99


def test_orthogonal_query_does_not_clear_the_bid_guard():
    """The caller only raises the bid when score > 0.0, so orthogonal must not."""
    core = _core(enable_retrieval=True)
    state = torch.zeros(4)
    state[0] = 1.0
    _store(core, state)
    query = torch.zeros(4)
    query[1] = 1.0
    assert core.get_similar_experiences(query, k=1)[0]["score"] <= 0.0


def test_scores_are_ordered_best_first():
    core = _core(enable_retrieval=True)
    near = torch.tensor([1.0, 0.9, 0.0, 0.0])
    far = torch.tensor([0.0, 0.0, 1.0, 1.0])
    _store(core, far)
    _store(core, near)
    results = core.get_similar_experiences(torch.tensor([1.0, 1.0, 0.0, 0.0]), k=2)
    assert results[0]["score"] > results[1]["score"]


def test_dimension_mismatch_scores_zero_and_does_not_raise():
    """Different dimensions mean different spaces; do not invent a comparison."""
    core = _core(enable_retrieval=True)
    _store(core, torch.ones(768))
    results = core.get_similar_experiences(torch.ones(801), k=1)
    assert results[0]["score"] == 0.0


def test_empty_memory_returns_empty_list():
    assert _core(enable_retrieval=True).get_similar_experiences(torch.ones(8), k=1) == []


def test_return_shape_is_unchanged():
    """train_rlhf.py:1060 reads `score`; the existing memory test reads all three."""
    core = _core(enable_retrieval=True)
    _store(core, torch.ones(8))
    match = core.get_similar_experiences(torch.ones(8), k=1)[0]
    for key in ("id", "score", "metadata"):
        assert key in match


def test_k_bounds_the_result_count():
    core = _core(enable_retrieval=True)
    for _ in range(5):
        _store(core, torch.rand(8))
    assert len(core.get_similar_experiences(torch.rand(8), k=3)) == 3


# --- why the phi gate does not block this ---------------------------------------------

def test_experiences_are_stored_below_the_attention_threshold():
    """The measured phi max is ~1.3e-02 against a 0.7 gate, so this is the live case.

    If this ever fails, the phi gate DOES block retrieval and the whole repair is void.
    """
    core = _core(enable_retrieval=True)
    _store(core, torch.ones(8), attention=0.001)
    assert len(core.recent_experiences) == 1
    assert core.get_similar_experiences(torch.ones(8), k=1)[0]["score"] > 0.99
