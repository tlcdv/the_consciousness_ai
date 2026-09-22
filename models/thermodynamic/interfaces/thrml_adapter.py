"""
Protected wrapper around Extropic's open-source THRML sampler (JAX, Python 3.10+).

THRML is optional and never in requirements.txt. Every function imports it on call;
without it, the call raises ImportError naming the missing package. No function
returns a fallback sample.

Convention, measured against exact enumeration on 2026-09-22 (THRML 0.1.4):
THRML's IsingEBM energy is E(s) = -(sum_i b_i s_i + sum_edges w_ij s_i s_j), the same
as ising_energy here, so couplings and biases pass through unchanged. THRML returns
spins as booleans (True = +1).

This samples in software on CPU or GPU. It does not reach Extropic hardware.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import torch

from models.thermodynamic.p_bit_emulator import _check_couplings, greedy_coloring


def thrml_available() -> bool:
    """True when both thrml and jax can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in ("thrml", "jax"))


def _require_thrml():
    if not thrml_available():
        raise ImportError("thrml (and jax) are not installed; they need Python 3.10+: pip install thrml")
    import jax
    import jax.numpy as jnp
    import thrml
    from thrml import models as thrml_models

    return jax, jnp, thrml, thrml_models


def _build_program(J: torch.Tensor, b: torch.Tensor, beta: float):
    jax, jnp, thrml, thrml_models = _require_thrml()
    nodes = [thrml.SpinNode() for _ in range(J.shape[0])]
    pairs = torch.nonzero(torch.triu(J, diagonal=1)).tolist()
    edges = [(nodes[i], nodes[j]) for i, j in pairs]
    weights = jnp.array([J[i, j].item() for i, j in pairs])
    model = thrml_models.IsingEBM(nodes, edges, jnp.array(b.tolist()), weights, jnp.array(beta))
    free_blocks = [thrml.Block([nodes[k] for k in members.tolist()]) for members in greedy_coloring(J)]
    program = thrml_models.IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    return model, program, free_blocks, nodes


def sample_with_thrml(
    J: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    n_chains: int,
    n_samples: int,
    n_warmup: int,
    steps_per_sample: int,
    seed: int,
) -> torch.Tensor:
    """Block Gibbs samples from THRML as +-1, shape [n_chains, n_samples, n]."""
    _check_couplings(J)
    jax, _, thrml, thrml_models = _require_thrml()
    model, program, free_blocks, nodes = _build_program(J, b, beta)
    init_key, sample_key = jax.random.split(jax.random.key(seed))
    init = thrml_models.hinton_init(init_key, model, free_blocks, (n_chains,))
    schedule = thrml.SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)
    run = jax.vmap(lambda key, state: thrml.sample_states(key, program, schedule, state, [], [thrml.Block(nodes)]))
    spins = np.asarray(run(jax.random.split(sample_key, n_chains), init)[0])
    return torch.from_numpy(2.0 * spins.astype(np.float64) - 1.0)
