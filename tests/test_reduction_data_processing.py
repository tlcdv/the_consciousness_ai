"""
Reducing a representation cannot create content, and binning cannot recover it.

These tests are the Python half of a pair. The other half is a machine checked
proof in the companion Lean specifications, where the same statements are
theorems rather than tested claims:

    FormalSpecs.Reduction.content_map_le           a readout never increases content
    FormalSpecs.Reduction.content_binning_le       binning never increases it further
    FormalSpecs.Reduction.content_binning_eq_zero  a contentless readout stays contentless
    FormalSpecs.Reduction.content_map_id           the bound is attained

The Lean proofs rest on the data processing inequality for the Kullback-Leibler
divergence, and depend on no axiom beyond propext, Classical.choice, Quot.sound.

Why both halves exist: the Lean file proves statements about a model written by
hand. It does not read this repository. These tests assert the same properties of
the code that actually runs, so a divergence between the model and the code fails
a test instead of going unnoticed. This narrows the gap. It does not close it.
Passing tests are evidence, not proof of equivalence.

Verification approach: these inequalities are not seed dependent, so each is
checked over many independent random cases with fixed seeds, alongside closed
form controls (the identity readout, total collapse) whose values are known in
advance. Every assertion RAISES on mismatch rather than printing.

What these tests do NOT show: that any scalar logged by this project is
contentless. That claim is empirical and rests on its own measurements. What is
established here is only that IF a readout is contentless, THEN no binning of it
recovers anything.
"""
import numpy as np
import pytest

from models.evaluation.effective_information import effective_information_from_tpm


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in bits. Both arguments must be strictly positive."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _pushforward(p: np.ndarray, groups: np.ndarray, n_out: int) -> np.ndarray:
    """Apply a readout: send state i to bin groups[i], then sum the mass.

    This is the discrete case of `Measure.map` in the Lean file.
    """
    out = np.zeros(n_out, dtype=np.float64)
    for i, g in enumerate(groups):
        out[g] += p[i]
    return out


def _random_pair(rng, n):
    """Two strictly positive distributions over n states."""
    return rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))


@pytest.mark.parametrize("seed", range(20))
def test_a_readout_never_increases_content(seed):
    """KL(f#p || f#q) <= KL(p || q) for every readout f.

    Pairs with FormalSpecs.Reduction.content_map_le.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 17))
    n_out = int(rng.integers(2, n))
    p, q = _random_pair(rng, n)
    groups = rng.integers(0, n_out, size=n)

    before = _kl(p, q)
    after = _kl(_pushforward(p, groups, n_out), _pushforward(q, groups, n_out))

    assert after <= before + 1e-12, (
        "a readout increased content: %r became %r" % (before, after))


@pytest.mark.parametrize("seed", range(20))
def test_binning_a_readout_never_increases_content(seed):
    """Binning is post processing, so it cannot increase content either.

    Pairs with FormalSpecs.Reduction.content_binning_le.
    """
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(6, 21))
    n_mid = int(rng.integers(3, n))
    n_bins = int(rng.integers(2, n_mid))
    p, q = _random_pair(rng, n)

    readout = rng.integers(0, n_mid, size=n)
    p_r = _pushforward(p, readout, n_mid)
    q_r = _pushforward(q, readout, n_mid)

    binning = rng.integers(0, n_bins, size=n_mid)
    p_b = _pushforward(p_r, binning, n_bins)
    q_b = _pushforward(q_r, binning, n_bins)

    assert _kl(p_b, q_b) <= _kl(p_r, q_r) + 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_a_contentless_readout_stays_contentless_under_every_binning(seed):
    """The statement that settles whether a finer binning is worth building.

    When a readout carries nothing, its content is zero, and every binning of it
    is still zero. A finer binning resolves more states, and each one of them is
    a state of nothing.

    Pairs with FormalSpecs.Reduction.content_binning_eq_zero.
    """
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(6, 17))
    n_mid = int(rng.integers(2, n))

    # A readout is contentless exactly when the two stimuli push forward to the
    # same distribution. Collapsing every state onto one value builds that case.
    p, q = _random_pair(rng, n)
    readout = np.zeros(n, dtype=int)
    p_r = _pushforward(p, readout, n_mid)
    q_r = _pushforward(q, readout, n_mid)
    assert _kl(p_r, q_r) == pytest.approx(0.0, abs=1e-12)

    for n_bins in range(2, n_mid + 3):
        binning = rng.integers(0, n_bins, size=n_mid)
        p_b = _pushforward(p_r, binning, n_bins)
        q_b = _pushforward(q_r, binning, n_bins)
        assert _kl(p_b, q_b) == pytest.approx(0.0, abs=1e-12), (
            "binning into %d bins recovered content from nothing" % n_bins)


@pytest.mark.parametrize("seed", range(10))
def test_the_identity_readout_loses_nothing(seed):
    """The bound is attained, so the tests above are not vacuous.

    Without this they would be consistent with "every reduction destroys
    everything", which is false and is not what the Lean file claims.

    Pairs with FormalSpecs.Reduction.content_map_id.
    """
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(4, 17))
    p, q = _random_pair(rng, n)
    identity = np.arange(n)

    assert _kl(_pushforward(p, identity, n),
               _pushforward(q, identity, n)) == pytest.approx(_kl(p, q), rel=1e-12)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_binning_a_degenerate_tpm_cannot_manufacture_effective_information(n):
    """The corollary, applied to the function this project actually calls.

    A TPM whose rows are all equal is the degenerate case: every cause drives the
    same effect, so a transition carries nothing about which state it came from.
    Its degeneracy corrected EI is zero. Coarse graining the effect states keeps
    the rows equal, so it stays zero however fine or coarse the bins are.

    This is content_binning_eq_zero read through Hoel's EI, which the module
    docstring of effective_information.py records as the average of
    KL(row_i || mean row).
    """
    rng = np.random.default_rng(4000 + n)
    row = rng.dirichlet(np.ones(n))
    tpm = np.tile(row, (n, 1))

    assert effective_information_from_tpm(
        tpm, degeneracy_corrected=True) == pytest.approx(0.0, abs=1e-12)

    for n_bins in range(2, n + 1):
        groups = rng.integers(0, n_bins, size=n)
        binned = np.zeros((n, n_bins), dtype=np.float64)
        for j, g in enumerate(groups):
            binned[:, g] += tpm[:, j]
        assert effective_information_from_tpm(
            binned, degeneracy_corrected=True) == pytest.approx(0.0, abs=1e-12), (
            "binning into %d bins manufactured EI from a degenerate TPM" % n_bins)
