"""The choice head of step P2 of design P (docs/decisions/2026_11_22_workspace_choice_mode.md).

The plain value head of step P1 reads one input row of 256 values and gives a value to slot 0 (the
candidate with the higher salience) and to slot 1. Before its first layer it normalizes each row over
its 256 values with no weight and no bias (the owner decision on the scaling in the rules). It explores
by a fixed share of random choices from a numpy generator of its own, and it learns with Adam from the
outcome of a choice. The rule `chosen` (the default) is the mean squared error of the output of the
chosen slot against the outcome. Check 2b of the rules compares it with two other rules, `both` and
`hopeful` (section "Owner decision before work 4, and the rules of check 2b" and the section after it).

The head draws no number from a random stream of the agent. Its initial weights come from the CPU
generator of torch inside a save and restore of that state, and no CUDA generator is seeded. An update
draws no number.

This file holds the head, its exploration and its updates. Check 2 of the rules uses the same code. It
also holds the live mode of work 4 (`LiveChoiceHead`): the slots of the candidates, the window of the
outcome, the 5 updates after a closed choice and the record of a run. The live mode uses the rule `chosen`
only (owner decision before work 4). The wiring to the training loop is in scripts/training/train_rlhf.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.value_gate import FOOD, OUTCOME_STEPS

INPUT_DIM = 256
HIDDEN = 128                 # ac_hidden of StandardActorCritic
EXPLORATION_SHARE = 0.25
UPDATES_PER_CHOICE = 5
LEARNING_RATE = 3e-4         # the default of both policy classes, as in step P1
NORM_EPS = 1e-5
RULES = ("chosen", "both", "hopeful")
DEFAULT_RULE = "chosen"
HOPEFUL_BIAS = 1.0           # rule (b): added to each bias of the last layer, so the outputs start near 1
LAST_LAYER = 4               # the index of the last linear layer in the body


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Each row to mean 0 and variance 1 over its last dimension, with no weight and no bias. A row of
    zeros stays zero."""
    return F.layer_norm(rows, (rows.shape[-1],), weight=None, bias=None, eps=NORM_EPS)


def plain_body(input_dim: int = INPUT_DIM) -> nn.Sequential:
    """The body of StandardActorCritic with one linear layer of 2 outputs, the plain value head of step P1."""
    return nn.Sequential(nn.Linear(input_dim, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, 2))


def seeded_body(seed: int, input_dim: int = INPUT_DIM) -> nn.Sequential:
    """The body with the default initial weights of the library, from `seed` on the CPU generator of
    torch only. The state of that generator is put back after, and no CUDA generator is seeded."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return plain_body(input_dim)


def exploring_choices(outputs: np.ndarray, rng: np.random.Generator,
                      share: float = EXPLORATION_SHARE) -> Tuple[np.ndarray, np.ndarray]:
    """One choice for each row of outputs, in the order of the rows. Each choice takes rng.random() and
    then rng.integers(0, 2), whatever the outputs are. When the first number is below the share, the
    choice is an exploration draw and takes the slot of the second number. Else it takes the slot with
    the larger output, and slot 0 when the two are equal. Returns the chosen slots and the draws."""
    chosen = np.empty(len(outputs), dtype=np.int64)
    explored = np.empty(len(outputs), dtype=bool)
    for index, (first, second) in enumerate(np.asarray(outputs)):
        uniform, slot = rng.random(), int(rng.integers(0, 2))
        explored[index] = uniform < share
        chosen[index] = slot if explored[index] else (0 if first >= second else 1)
    return chosen, explored


def as_rows(rows) -> torch.Tensor:
    """A float32 tensor of rows on the CPU. One row of 256 values becomes a batch of one."""
    tensor = torch.as_tensor(np.asarray(rows, dtype=np.float32))
    return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor


def chosen_slot_loss(outputs: torch.Tensor, chosen: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """The mean squared error of the output of the chosen slot against the outcome."""
    return nn.MSELoss()(outputs.gather(1, chosen.long().unsqueeze(1)), outcome)


def both_slots_targets(chosen: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Rule (a): the target of the chosen output is the outcome, the target of the other output is 1 minus
    the outcome. `outcome` has one column."""
    chosen_is_0 = (chosen.long() == 0).unsqueeze(1).to(outcome.dtype)
    return torch.cat([chosen_is_0 * outcome + (1 - chosen_is_0) * (1 - outcome),
                      chosen_is_0 * (1 - outcome) + (1 - chosen_is_0) * outcome], dim=1)


def both_slots_loss(outputs: torch.Tensor, chosen: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Rule (a): the mean squared error over the two outputs against the targets of both_slots_targets."""
    return nn.MSELoss()(outputs, both_slots_targets(chosen, outcome))


def add_hopeful_bias(body: nn.Sequential) -> None:
    """Rule (b): 1 added to each of the two biases of the last layer. It draws no number."""
    with torch.no_grad():
        body[LAST_LAYER].bias.add_(HOPEFUL_BIAS)


class ChoiceHead:
    """The plain value head with its normalization, its exploration generator, its optimizer and its
    learning rule. `chosen` trains the output of the chosen slot, `both` also trains the other output
    toward 1 minus the outcome, and `hopeful` is `chosen` with 1 added to the biases of the last layer."""

    def __init__(self, head_seed: int, exploration_seed, share: float = EXPLORATION_SHARE,
                 learning_rate: float = LEARNING_RATE, input_dim: int = INPUT_DIM, rule: str = DEFAULT_RULE):
        if rule not in RULES:
            raise ValueError("the rule of the choice head is %r, not one of %s" % (rule, ", ".join(RULES)))
        self.rule = rule
        self.body = seeded_body(head_seed, input_dim)
        if rule == "hopeful":
            add_hopeful_bias(self.body)
        self.rng = np.random.default_rng(exploration_seed)
        self.share = share
        self.optimizer = torch.optim.Adam(self.body.parameters(), lr=learning_rate)

    def forward_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """The two outputs of each row: the body applied to the normalized row."""
        return self.body(normalize_rows(rows))

    def outputs(self, rows) -> np.ndarray:
        """The two outputs of each row as float32 numbers, with no gradient."""
        with torch.no_grad():
            return self.forward_rows(as_rows(rows)).numpy()

    def preferences(self, rows) -> np.ndarray:
        """Output 0 minus output 1 for each row, the preference of the score of the rules."""
        outputs = self.outputs(rows)
        return (outputs[:, 0] - outputs[:, 1]).astype(np.float64)

    def choose(self, rows) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The chosen slot of each row in the order of the rows, which choices were exploration draws,
        and the outputs that the choices read."""
        outputs = self.outputs(rows)
        chosen, explored = exploring_choices(outputs, self.rng, self.share)
        return chosen, explored, outputs

    def update(self, rows, chosen, outcome, steps: int = 1) -> float:
        """`steps` Adam updates of the loss of the rule on these rows. Returns the loss before the first
        update. A loss that is not finite raises FloatingPointError."""
        rows = as_rows(rows)
        chosen = torch.as_tensor(np.asarray(chosen), dtype=torch.long)
        outcome = torch.as_tensor(np.asarray(outcome, dtype=np.float32)).unsqueeze(1)
        first_loss = None
        loss_of_rule = both_slots_loss if self.rule == "both" else chosen_slot_loss
        for _ in range(steps):
            loss = loss_of_rule(self.forward_rows(rows), chosen, outcome)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("the loss of the choice head is %r, not finite" % loss.item())
            first_loss = loss.item() if first_loss is None else first_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return first_loss

    def loss(self, rows, chosen, outcome) -> float:
        """The loss of the rule on these rows, with no gradient and no update."""
        rows = as_rows(rows)
        chosen = torch.as_tensor(np.asarray(chosen), dtype=torch.long)
        outcome = torch.as_tensor(np.asarray(outcome, dtype=np.float32)).unsqueeze(1)
        loss_of_rule = both_slots_loss if self.rule == "both" else chosen_slot_loss
        with torch.no_grad():
            return float(loss_of_rule(self.forward_rows(rows), chosen, outcome).item())


# The live mode of work 4: the slots, the window of the outcome and the record of one run. The wiring to
# the training loop is `_wire_choice_head` in scripts/training/train_rlhf.py.

HEAD_SEED_OFFSET = 2000003        # the head seed is the run seed + this
EXPLORATION_SEED_OFFSET = 3000005  # the exploration seed is the run seed + this


def candidate_slots(candidates: Dict[int, float]) -> Tuple[List[int], bool]:
    """The bands in the order of the slots, and the tie. Slot 0 is the band with the higher salience.
    With equal salience the lower band is slot 0, and the tie is True. More than two candidates raise
    ValueError, because the binding of two bands gives two at most."""
    if not 1 <= len(candidates) <= 2:
        raise ValueError("the choice head needs one or two candidates, got %d" % len(candidates))
    bands = sorted(candidates, key=lambda band: (-candidates[band], band))
    tie = len(bands) == 2 and candidates[bands[0]] == candidates[bands[1]]
    return bands, bool(tie)


def input_row(vector) -> np.ndarray:
    """A float32 copy on the CPU of the vector that the policy received, as one row of 256 values."""
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().float().cpu().numpy()
    row = np.asarray(vector, dtype=np.float32).reshape(-1).copy()
    if row.size != INPUT_DIM:
        raise ValueError("the choice head reads %d values, got %d" % (INPUT_DIM, row.size))
    return row


class OutcomeWindow:
    """The binary outcome of the value gate: 1 when a step reward above FOOD comes within OUTCOME_STEPS
    steps after the choice, 0 at the last step of the window without it. Each observed reward counts."""

    def __init__(self, food: float = FOOD, outcome_steps: int = OUTCOME_STEPS):
        self.food, self.outcome_steps, self.steps = food, outcome_steps, 0

    def observe(self, reward: float) -> Optional[float]:
        """The outcome when this reward closes the window, else None."""
        self.steps += 1
        if reward > self.food:
            return 1.0
        return 0.0 if self.steps >= self.outcome_steps else None


class LiveChoiceHead:
    """The head of one run with its open choice, its outcome window and its record. Once in each episode
    of the value gate (a step without sound starts the next one), at the first step with candidates, it
    chooses a slot, and the choice is kept until the next step without sound."""

    def __init__(self, run_seed: int):
        self.head_seed = run_seed + HEAD_SEED_OFFSET
        self.exploration_seed = run_seed + EXPLORATION_SEED_OFFSET
        self.head = ChoiceHead(self.head_seed, self.exploration_seed)
        self.choice: Optional[int] = None
        self.entries: List[dict] = []
        self.rows: List[np.ndarray] = []
        self._open = None                 # (entry, row, window) of the open choice
        self.episode: Optional[int] = None
        self.boundary = 0                 # the length of the map history at the last reset of the room

    def settings(self) -> dict:
        return {"head": "plain", "rule": self.head.rule, "share": self.head.share,
                "updates_per_choice": UPDATES_PER_CHOICE, "learning_rate": LEARNING_RATE,
                "head_seed": self.head_seed, "exploration_seed": self.exploration_seed}

    def choose(self, candidates: Dict[int, float], row: np.ndarray, step: int, episode) -> int:
        """The band to follow. With two candidates the head draws its two numbers and chooses a slot.
        With one candidate there is no draw, and the head follows it."""
        bands, tie = candidate_slots(candidates)
        entry = {"episode": episode, "step": int(step), "salience": {str(b): float(s) for b, s in candidates.items()},
                 "slots": bands, "tie": tie, "one_candidate": len(bands) == 1, "outputs": None,
                 "greedy_slot": None, "explored": False, "chosen_slot": 0, "choice": bands[0], "outcome": None,
                 "steps": 0, "loss_before": None, "loss_after": None, "updates": 0}
        if len(bands) == 2:
            chosen, explored, outputs = self.head.choose(row)
            entry.update(outputs=[float(v) for v in outputs[0]], greedy_slot=0 if outputs[0, 0] >= outputs[0, 1] else 1,
                         explored=bool(explored[0]), chosen_slot=int(chosen[0]), choice=bands[int(chosen[0])])
        self.entries.append(entry)
        self.rows.append(np.asarray(row, dtype=np.float32))
        self._open = (entry, self.rows[-1], OutcomeWindow())
        self.choice = entry["choice"]
        return self.choice

    def observe_reward(self, reward: float) -> None:
        """A reward of the room for the open choice. It can close the choice with its outcome."""
        if self._open is None:
            return
        outcome = self._open[2].observe(float(reward))
        if outcome is not None:
            self._close(outcome)

    def new_episode(self) -> None:
        """A step without sound: an open choice closes with 0, and the next step with candidates chooses."""
        if self._open is not None:
            self._close(0.0)
        self.choice = None

    def _close(self, outcome: float) -> None:
        entry, row, window = self._open
        self._open = None
        entry.update(outcome=float(outcome), steps=window.steps)
        if entry["one_candidate"]:
            return
        slot, target = [entry["chosen_slot"]], [float(outcome)]
        entry["loss_before"] = self.head.update(row, slot, target, steps=UPDATES_PER_CHOICE)
        entry["loss_after"] = self.head.loss(row, slot, target)
        entry["updates"] = UPDATES_PER_CHOICE

    def record(self) -> dict:
        """The content of choice_head.json. A choice that is still open has no outcome."""
        if self._open is not None:
            self._open[0]["steps"] = self._open[2].steps
        return {"settings": self.settings(), "choices": self.entries}

    def state(self) -> dict:
        """The content of choice_head.pt: the weights and the state of the optimizer."""
        return {"body": self.head.body.state_dict(), "optimizer": self.head.optimizer.state_dict()}
