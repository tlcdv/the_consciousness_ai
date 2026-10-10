# Thermodynamic transduction, gate v3 replication

**Verdict. The gate v3 PASS did not replicate. FAILED on `workspace_broadcast` and `obs_map`. UNTESTABLE on
`tectum_content` and `z_state`. No stream passed on the unused stimulus seeds.**

| Stream | Run 1 (stimulus 142 to 144) | Run 2, this replication (stimulus 242 to 244) |
|---|---|---|
| workspace_broadcast | PASSED | **FAILED**. K2 fails at model 42 (gap -0.0550, margin -0.05) |
| obs_map | PASSED | **FAILED**. K2 fails at models 42 and 43 (gaps -0.0617, -0.0783) |
| tectum_content | UNTESTABLE | UNTESTABLE. K1 fails at model 44 |
| z_state | UNTESTABLE | UNTESTABLE. K1 fails at models 42 and 44 |

The gate text, the code, the three checkpoints, beta, sweeps and permutation count were identical to run 1. The probe
file had no change between the v3 pre-statement (`e213c02`) and this run. Only the stimulus sequences differ. The result
document of run 1 (`thermodynamic_transduction_gate_v3_2026_10.md`) warned that the pass was narrow. This run confirms it.
Run 1 stands as recorded, and it is no longer a basis for any claim.

Source. `runs/thermo_cost/gate_v3_replication.json` (complete, 3 seeds, 100 permutations) and its log, written on
2026-10-10. The run took about 4.5 hours, against about 65 minutes for the earlier runs. The cause is unknown. Another
Python job was running on the machine at the same time. The numbers do not depend on speed.

## Accuracy per stream and seed

Columns. ridge = linear readout (informational). quant = sign of the projected vector, no noise, no settling.
noisy = Poisson spikes without settling. settled = the full pipeline. control = settled at beta 0.05.
Nulls are the p95 over 100 shuffles. K2 holds if settled is at least quant minus 0.05.

| Stream | Model / stimulus | ridge | quant | noisy | settled | control | null p95 quant | null p95 settled | settled - quant | Gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| tectum_content | 42 / 242 | 0.287 | 0.268 | 0.270 | 0.248 | 0.158 | 0.237 | 0.222 | -0.0200 | none |
| tectum_content | 43 / 243 | 0.508 | 0.487 | 0.477 | 0.397 | 0.177 | 0.225 | 0.219 | -0.0900 | K2 |
| tectum_content | 44 / 244 | 0.108 | 0.093 | 0.095 | 0.143 | 0.155 | 0.225 | 0.214 | +0.0500 | K1, K3 |
| workspace_broadcast | 42 / 242 | 0.358 | 0.358 | 0.330 | 0.303 | 0.182 | 0.225 | 0.230 | -0.0550 | K2 |
| workspace_broadcast | 43 / 243 | 0.508 | 0.400 | 0.405 | 0.388 | 0.157 | 0.234 | 0.217 | -0.0117 | none |
| workspace_broadcast | 44 / 244 | 0.300 | 0.267 | 0.282 | 0.225 | 0.153 | 0.225 | 0.220 | -0.0417 | none |
| obs_map | 42 / 242 | 0.542 | 0.358 | 0.423 | 0.297 | 0.157 | 0.242 | 0.205 | -0.0617 | K2 |
| obs_map | 43 / 243 | 0.633 | 0.458 | 0.535 | 0.380 | 0.167 | 0.225 | 0.210 | -0.0783 | K2 |
| obs_map | 44 / 244 | 0.450 | 0.242 | 0.368 | 0.288 | 0.152 | 0.225 | 0.207 | +0.0467 | none |
| z_state | 42 / 242 | 0.267 | 0.183 | 0.208 | 0.160 | 0.153 | 0.225 | 0.197 | -0.0233 | K1, K3 |
| z_state | 43 / 243 | 0.542 | 0.383 | 0.432 | 0.295 | 0.158 | 0.225 | 0.202 | -0.0883 | K2 |
| z_state | 44 / 244 | 0.120 | 0.150 | 0.155 | 0.147 | 0.163 | 0.222 | 0.202 | -0.0033 | K1, K3 |

## What the six draws say together

Descriptive only. The pooling is not a gate, and the two runs are not independent draws of one pre-stated plan, because run 1
informed the choice of the gate that run 2 repeats.

| Stream | Settled minus quant, 6 draws (142 to 144, then 242 to 244) | Mean | Draws failing K2 |
|---|---|---|---|
| tectum_content | -0.025, -0.050, +0.022, -0.020, -0.090, +0.050 | -0.019 | 1 of 6 |
| workspace_broadcast | -0.048, +0.003, -0.032, -0.055, -0.012, -0.042 | -0.031 | 1 of 6 |
| obs_map | -0.050, -0.018, -0.002, -0.062, -0.078, +0.047 | -0.027 | 2 of 6 |
| z_state | -0.052, -0.065, +0.020, -0.023, -0.088, -0.003 | -0.035 | 3 of 6 |

- The settled arm is below the noise-free binarised arm in most draws, with a mean gap of 0.02 to 0.04 accuracy points, and
  the gap varies by 0.06 to 0.14 points between draws within a stream. A margin of 0.05 sits inside that variation. So a pass or a fail at
  this margin depends on the stimulus draw, and neither is a stable property of a stream.
- The data fit a small real cost of settling of about 0.03 points, below the margin on average and above it in 7 of the 24
  stream and draw cells. I did not test that reading with a gate.
- In every cell the beta 0.05 control stayed at or below the null p95 (0.148 to 0.185 over both runs), and in every cell where
  the binarised readout was above its null (K1), the settled arm stayed above its own null (K3). The class survives settling at well above chance. How much of the
  binarised readout's accuracy survives is the part that did not hold at the stated margin.
- `tectum_content` and `z_state` fail the class test at model 44 in every run (v2, v3 run 1, v3 run 2), and `z_state` also at
  model 42 in this run. The cause is unknown.
- The noise-free binarised readout is far below the ridge readout on `obs_map` (0.24 to 0.46 against 0.45 to 0.63 here). Even
  a pass would not show that the original vector survives.

## Quantities that are not findings

- `fep_free_energy` change varies with normalisation and component count and is not comparable across streams.
- The GPU to p-bit energy ratio is arithmetic on assumed constants (1 W, 50 MHz, 6.7e10 FLOP per joule). It is not a
  measurement of any chip.
- `pbit_energy` at the last sweep reflects stored-state energy. It says nothing about the class.

## Standing of the instruments

`thermodynamic_entropy`, `pbit_energy` and `fep_free_energy` stay UNPROVEN. None was shown to depend on the stimulus class.
0 instruments are TRUSTED. No rubric entry or indicator changes. This result says nothing about consciousness or Phi.

## Consequence for publication

No number from this track goes on the website. The result does not support the claim that the verified representations
survive p-bit transduction at the stated tolerance. It also does not refute a small cost of settling. A public statement, if
the owner wants one, can describe the method and the mixed outcome: a pass that did not replicate. It needs no number that
is not in this document or in the earlier ones.

## What would change this

1. **A different question, pre-stated, with an estimate and not a pass line.** For example, report the settled minus
   binarised gap per stream with an interval over many stimulus draws, and decide in advance what size of gap counts as
   acceptable. This needs more draws than 6 to bound a gap whose spread is up to 0.14.
2. **Fresh checkpoints.** Three newly trained models, to separate model from stimulus draw. About 3 hours of serial training.
3. **Understanding model 44.** Why `tectum_content` and `z_state` do not carry the class there.
4. **A readout with more than one bit per dimension**, to test whether binarisation is the main loss.
5. **Stopping.** Three gates, three versions and six draws have not produced a stable pass. Further gates chosen after
   seeing results lean toward searching for a pass. The package, the tests, the blueprint and these records stay as the
   outcome of this track.
