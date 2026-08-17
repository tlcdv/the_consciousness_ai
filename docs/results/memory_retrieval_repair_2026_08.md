# The memory bid was disconnected. Reconnecting it moved it from one constant to another

**The stub is fixed and the bid is still degenerate.** Memory retrieval queried a
placeholder whose score was hardcoded to 0.0, so the workspace memory bid sat at its 0.1
floor at 24,000 of 24,000 steps. With retrieval searching the experiences that were
already being stored, the bid now sits at or within 1e-07 of its 0.6 CAP on about 99
percent of steps, at 3 seeds.

The competition did not change and was not expected to. Vision bids exactly 1.0 and the
memory bid caps at 0.6.

What the repair did produce is a measurement that was not the goal: **the broadcast points
in the same direction at essentially every step**, cosine 0.99999976 to 1.0 against its
best recent match. That bears directly on the open question of whether the workspace
signal is alive.

3 seeds, 40 episodes, 8000 steps each, dmts, `--enable-audio --enable-mock-semantic`.

## The defect

`models/memory/memory_core.py` defined `PineconeIndexStub`, whose `upsert` is `pass` and
whose `query` returns one match with `score = 0.0` hardcoded. The workspace bid at
`train_rlhf.py:1061` guards on `similar[0]["score"] > 0.0`, so the branch could never fire.

The experiences were never missing. `store_experience` appends every one to
`recent_experiences`, and that append sits OUTSIDE the `attention_level >= 0.7` gate, so
it runs on every step. Retrieval searched somewhere else.

This also dissolved a second blocker reported earlier the same day. The phi gate is
measured unreachable (phi max 6.9e-03 to 1.3e-02 against a 0.7 threshold, 0 of 8000 steps
passing at every seed), but it only ever controlled the no-op upsert.

## The repair

`--enable-memory-retrieval`, default off, baseline bit-identical. Retrieval searches
`recent_experiences` by cosine similarity on the `state` field, not the `vector` field:
`_create_memory_vector` builds `cat([state, action, emotion])` while every caller's query
is a bare state, so scoring against `vector` would compare quantities that are not
comparable.

Suite 962 passed, 5 skipped. All 28 pre-existing memory tests pass unchanged.

## Result: the bug is fixed, the bid is not repaired

| seed | flag OFF distinct | flag ON distinct | at the 0.6 cap | at the 0.1 floor |
|---|---|---|---|---|
| 42 | 1 | 6 | 4194 of 8000 (52.4%) | 61 |
| 43 | 1 | 5 | 3919 of 8000 (49.0%) | 44 |
| 44 | 1 | 5 | 4089 of 8000 (51.1%) | 58 |

The pre-stated gate asked only for more than one distinct value inside [0.1, 0.6]. On that
gate this is FIXED at 3 seeds.

**That gate was too weak, and the weakness was mine.** The same session applied a
non-degeneracy standard to the vision bid candidates in
`scripts/analysis/probe_bid_reduction_candidates.py`: at least 100 distinct values, span
at least 0.2, at most 1 percent of steps pinned at a bound. Against that standard the
repaired memory bid FAILS at all 3 seeds, on the distinct-value and pinned criteria. A bid
pinned at its ceiling half the time is no more informative than one pinned at its floor.

## Why: the broadcast has one direction

Inverting `bid = min(0.6, 0.1 + score * 0.5)` recovers the cosine similarity between the
query broadcast and its best match among recent stored broadcasts:

| seed | cosine >= 1.0 | cosine in between | range of the in-between values | cosine <= 0 |
|---|---|---|---|---|
| 42 | 4194 | 3745 | 0.99999976 to 0.99999994 | 61 |
| 43 | 3919 | 4037 | 0.99999982 to 0.99999994 | 44 |
| 44 | 4089 | 3853 | 0.99999982 to 0.99999994 | 58 |

At every step except roughly 50 per run, some recent broadcast is identical to the current
one to seven decimal places. The steps at the floor are the small minority where no
comparable experience exists yet.

Cosine measures direction only, and the magnitude is not equally frozen: `broadcast_mag`
has a standard deviation of 8.9e-03 to 2.1e-02 over the same runs. So the broadcast
changes length and does not change direction.

## Winners, reported not gated

| seed | flag OFF | flag ON |
|---|---|---|
| 42 | vision 7922, semantic 58 | vision 7932, semantic 47 |
| 43 | vision 7985, semantic 1 | vision 7996 |
| 44 | vision 7918, semantic 53 | vision 7936, semantic 46 |

Unchanged, as predicted before the runs. Vision at 1.0 against a memory cap of 0.6 makes
this arithmetic rather than an empirical question.

## What this changes

**The "it is only a wiring bug" defence for GWT-1 is gone.** Before this run, the constant
memory bid could be dismissed as a stub. It cannot now. With retrieval working correctly
against real stored data, the memory bid is still a constant, because the thing it
compares is itself constant in direction. That is a property of the system, not of a
placeholder.

**It is evidence on whether the workspace signal is alive.** The open question asks that
about the pre-discretization broadcast values. This does not answer it in the form asked,
which concerns the discretized workspace level. It does establish, from a different
direction and without any discretization involved, that the broadcast direction is
constant to 1e-07 across a run.

## What this does NOT settle

- **No indicator is re-scored.** GWT-1 stays IMPLEMENTED, the count stays 4/10/14.
- **The repair is not a fix for the competition** and must not be cited as one.
- **Cosine is direction only.** A constant direction with varying magnitude is not the
  same as a frozen broadcast, and the magnitude does vary.
- **The comparison is against the best of a bounded recent pool**, not against every past
  step. It shows the broadcast keeps revisiting one direction, not that all pairs are
  identical.
- **The default stays off.** Whether a bid pinned at 0.6 is better than one pinned at 0.1
  is not obvious, and flipping the default is a separate decision.

## Reproduce

```
python -m scripts.training.train_rlhf --env dmts --episodes 40 --seed 42 \
  --rssm-latent-mode continuous --capsule-workspace-source all_levels \
  --enable-audio --enable-mock-semantic --enable-memory-retrieval \
  --log-dir runs/memfix_s42
```
