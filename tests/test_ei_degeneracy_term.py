"""
The local EI formula omits Hoel's degeneracy term. These tests pin both formulas.

`compute_effective_information` computes `log2(n) - mean_i H(row_i)`. Hoel's EI is
`H(<TPM>) - mean_i H(row_i)`, where `<TPM>` is the average row, i.e. the effect
distribution under do(uniform). The first term is where degeneracy lives: when many
causes drive the same effect, that distribution concentrates and EI falls.

The legacy form fails in the UNSAFE direction. A system whose every state maps to one
effect scores log2(n), the maximum, where the correct answer is 0.

Clause 5 governs this file: these are deterministic analytic results with no seeds to
average, so they need closed-form controls and an independent derivation, verified in
code that RAISES on mismatch rather than printing. Both are here.
"""
import numpy as np
import pytest

from models.evaluation.effective_information import (
    compute_effective_information,
    effective_information_from_tpm,
)


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _ei_kl_form(tpm: np.ndarray) -> float:
    """Hoel's OTHER published form: (1/n) sum_i KL(row_i || <TPM>).

    An independent derivation, not a restatement of the implementation. If this
    disagrees with the entropy form, the reading of the source is wrong.
    """
    tpm = np.asarray(tpm, dtype=np.float64)
    mean_row = tpm.mean(axis=0)
    total = 0.0
    for row in tpm:
        mask = row > 0
        total += float(np.sum(row[mask] * np.log2(row[mask] / mean_row[mask])))
    return total / len(tpm)


def _permutation_tpm(n: int) -> np.ndarray:
    """Deterministic and non-degenerate: a bijection on states."""
    return np.roll(np.eye(n, dtype=np.float64), 1, axis=1)


def _collapse_tpm(n: int) -> np.ndarray:
    """Maximally degenerate: every cause drives the same effect."""
    tpm = np.zeros((n, n), dtype=np.float64)
    tpm[:, 0] = 1.0
    return tpm


def _uniform_tpm(n: int) -> np.ndarray:
    """Maximally noisy: the effect is independent of the cause."""
    return np.full((n, n), 1.0 / n, dtype=np.float64)


# --- independent derivation: the two published forms must agree ---------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_two_published_forms_agree(seed):
    """RAISES on mismatch. This is what licenses the entropy form as 'Hoel's EI'."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        n = int(rng.integers(3, 40))
        tpm = rng.random((n, n))
        tpm /= tpm.sum(axis=1, keepdims=True)
        entropy_form = effective_information_from_tpm(tpm, degeneracy_corrected=True)
        assert entropy_form == pytest.approx(_ei_kl_form(tpm), abs=1e-9), (
            "the entropy form and the KL form disagree, so the corrected formula is "
            "not Hoel's EI"
        )


# --- closed-form controls -----------------------------------------------------------

@pytest.mark.parametrize("n", [3, 8, 27, 243])
def test_permutation_gives_log2n_under_both_formulas(n):
    """No degeneracy, so the missing term is zero and the two formulas coincide."""
    tpm = _permutation_tpm(n)
    assert effective_information_from_tpm(tpm, True) == pytest.approx(np.log2(n), abs=1e-9)
    assert effective_information_from_tpm(tpm, False) == pytest.approx(np.log2(n), abs=1e-9)


@pytest.mark.parametrize("n", [3, 8, 27, 243])
def test_total_collapse_is_zero_corrected_and_maximal_legacy(n):
    """THE DEFECT, pinned as a closed form.

    Every cause drives one effect, so the cause tells you nothing about the effect and
    EI is exactly 0. The legacy formula returns log2(n), the largest value it can
    produce, for the least informative system there is.
    """
    tpm = _collapse_tpm(n)
    assert effective_information_from_tpm(tpm, degeneracy_corrected=True) == pytest.approx(0.0, abs=1e-12)
    assert effective_information_from_tpm(tpm, degeneracy_corrected=False) == pytest.approx(np.log2(n), abs=1e-12)


@pytest.mark.parametrize("n", [3, 8, 27])
def test_uniform_tpm_is_zero_under_both(n):
    """Maximal noise. Both formulas agree here, so the defect is degeneracy-specific
    rather than a general disagreement."""
    tpm = _uniform_tpm(n)
    assert effective_information_from_tpm(tpm, True) == pytest.approx(0.0, abs=1e-12)
    assert effective_information_from_tpm(tpm, False) == pytest.approx(0.0, abs=1e-12)


def test_partial_collapse_sits_between_the_extremes():
    """Half the states share an effect: corrected EI must drop, legacy must not move."""
    n = 8
    tpm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        tpm[i, (i // 2) % n] = 1.0
    corrected = effective_information_from_tpm(tpm, True)
    legacy = effective_information_from_tpm(tpm, False)
    assert 0.0 < corrected < np.log2(n)
    assert legacy == pytest.approx(np.log2(n), abs=1e-12)
    assert corrected < legacy


def test_corrected_never_exceeds_legacy():
    """Direction of the bias, which is what bounds every historical EI number.

    H(<TPM>) <= log2(n) always, so the omitted term can only INFLATE the legacy value.
    Every published EI figure is therefore an upper bound on the corrected one.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(3, 30))
        tpm = rng.random((n, n))
        tpm /= tpm.sum(axis=1, keepdims=True)
        assert (effective_information_from_tpm(tpm, True)
                <= effective_information_from_tpm(tpm, False) + 1e-12)


# --- the default must not have moved ------------------------------------------------

def test_default_path_is_bit_identical_to_the_legacy_formula():
    """Every historical number and every pre-registered threshold refers to this."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        num_states = int(rng.integers(3, 12))
        traj = rng.integers(0, num_states, size=200)
        from models.evaluation.effective_information import _build_tpm
        expected = float(np.log2(num_states) - np.mean(
            [_entropy(_build_tpm([traj], num_states)[i]) for i in range(num_states)]))
        assert compute_effective_information([traj], num_states) == pytest.approx(
            max(0.0, expected), abs=1e-12)


def test_why_the_observational_estimator_hid_this():
    """Laplace smoothing on sparse coverage makes the two formulas nearly agree.

    Pinned because it is the explanation for a defect that survived every previous
    audit, and because it predicts the gap OPENS on an interventional matrix. That
    prediction is what the interventional probe tests.
    """
    from models.evaluation.effective_information import _build_tpm

    n = 243
    frozen = [np.zeros(2000, dtype=np.int64)]      # one state visited, maximally degenerate
    smoothed = _build_tpm(frozen, n)
    legacy = effective_information_from_tpm(smoothed, False)
    corrected = effective_information_from_tpm(smoothed, True)

    # Sparse observation keeps 242 of 243 rows uniform, so the average row is nearly
    # uniform and the two formulas stay close despite total degeneracy.
    assert abs(legacy - corrected) < 0.05, (
        "the observational TPM no longer masks the missing term; if this changed, the "
        "explanation for why the defect went unnoticed needs revisiting"
    )

    # The same degeneracy expressed as a SHARP matrix, which is what intervention
    # produces, opens the gap to the full log2(n).
    sharp = _collapse_tpm(n)
    assert effective_information_from_tpm(sharp, False) - \
           effective_information_from_tpm(sharp, True) == pytest.approx(np.log2(n), abs=1e-9)
