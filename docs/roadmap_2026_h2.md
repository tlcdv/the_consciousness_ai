# Working roadmap 2026 H2: gated tracks toward measurable consciousness signatures

This is the operational companion to [roadmap.md](roadmap.md). The main roadmap records
phases and history; this document is the forward decision tree for the coming work
sessions. It exists because the project's recent history is a chain of pre-registered
bets that FAILED at their kill gates (Phi-1 x9 runs, reconstruction x2 targets,
value-equivalent world model, supervised latent-identity ceiling), and the honest way
to plan is to state, in advance, what each outcome redirects to.

## How to read this document

- Work is organized in **tracks** (A through E). Tracks A and B are independent and can
  interleave; C is blocked behind A; D and E are background tracks.
- Every experiment has a **gate**: a pre-stated pass/fail reading. FAILED is said first.
- Marks: **[USER GATE]** = do not open without an explicit owner decision.
  **[ZERO-COMPUTE]** = analysis of existing data only. Costs are wall-clock on the
  current machine (DMTS 100 episodes is ~50 minutes; the collapse-locus probe is
  ~5 minutes; the test suite is ~95 seconds). All heavy jobs run serially.
- Single seed = hypothesis. Three or more seeds before any headline number or any
  default-flag change. No numeric threshold in this document is new; gates reuse the
  established references (decode chance = 0.167, the obs_map control at ~1.0, the CE
  chance floor 2 ln 6 = 3.584).

## Current position (2026-07-05, all from committed verdicts)

- **The identity wall:** stimulus identity dies at the obs_map to z_state step of the
  RSSM and stays dead under reward training, frame reconstruction, obs_map
  reconstruction, a value-equivalent world model, and direct supervision at the locus
  ([latent_identity_ceiling_2026_07.md](results/latent_identity_ceiling_2026_07.md)).
  The discrete gumbel-softmax categorical latent, or its batch-1 online optimization,
  is the wall. Label-free contrastive objectives on this latent are ruled out.
- **The instrument problem:** most signature instruments read constants on the current
  agent ([signature_assessment_2026_07.md](results/signature_assessment_2026_07.md)):
  EI is a Laplace-floor artifact at both levels, GNW ignition is saturated, sync_R is
  objective-invariant, and RIIU/Levin/self-prediction log zeros (disabled modules).
  Phi is the one objective-sensitive signal (single seed).
- **The competence wall:** independent of perception, the RL loop did not learn DMTS
  even when identity was offline-decodable (the 2026-06-14/15 finding). Causal-efficacy
  evidence (section 13) needs both walls down.

Success remains judged by consciousness signatures (Butlin indicators, the
pre-registered substrate-independence test), never by task reward, and "conscious" is
never claimed. Phi-1 stays closed. Section-13 thresholds stay untouched.

---

## Track A: the identity wall (perception)

The question this track answers: can ANY training regime make the agent's integrated
latent carry stimulus identity, and if so, which