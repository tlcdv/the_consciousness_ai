"""Predicted band of each hypothesis for the broadcast swap of step P3, with a head of two slots.

The rules are in docs/decisions/2026_11_24_p3_swap_predictions.md. This module reads no run and computes no
measured number. It gives what each hypothesis predicts, the class of a swap, and whether a set of swaps
separates the three hypotheses. The training script does not import it.
"""
from dataclasses import dataclass
from typing import Iterable, Tuple

HYPOTHESES = ("ignore", "band", "slot")


@dataclass(frozen=True)
class Episode:
    """The target band of one episode and its two candidate bands, slot 0 first (higher salience)."""
    target_band: int
    slot_order: Tuple[int, ...]

    def __post_init__(self):
        if len(self.slot_order) != 2 or self.slot_order[0] == self.slot_order[1]:
            raise ValueError(f"slot_order needs two different bands, got {self.slot_order}")
        if self.target_band not in self.slot_order:
            raise ValueError(f"target band {self.target_band} is not a candidate of {self.slot_order}")


def _check_same_bands(recipient: Episode, donor: Episode) -> None:
    if set(recipient.slot_order) != set(donor.slot_order):
        raise ValueError(f"the swap has other bands ({recipient.slot_order} and {donor.slot_order}), "
                         "so it is in no class")


def donor_class(recipient: Episode, donor: Episode) -> str:
    """C0 (self replacement), C1, C2 or C3, from the targets and the salience orders."""
    _check_same_bands(recipient, donor)
    same_target = donor.target_band == recipient.target_band
    same_order = donor.slot_order == recipient.slot_order
    if same_target:
        return "C0" if same_order else "C3"
    return "C1" if same_order else "C2"


def predicted_band(hypothesis: str, recipient: Episode, donor: Episode) -> int:
    """The band that the hypothesis predicts when the recipient's head reads the donor broadcast."""
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown hypothesis {hypothesis!r}, expected one of {HYPOTHESES}")
    _check_same_bands(recipient, donor)
    if hypothesis == "ignore":
        return recipient.target_band
    if hypothesis == "band":
        return donor.target_band
    donor_slot = donor.slot_order.index(donor.target_band)
    return recipient.slot_order[donor_slot]


def separates(swaps: Iterable[Tuple[Episode, Episode]]) -> bool:
    """True when the three hypotheses predict three different sequences of bands over these swaps."""
    swaps = list(swaps)
    rows = {tuple(predicted_band(h, recipient, donor) for recipient, donor in swaps) for h in HYPOTHESES}
    return len(swaps) > 0 and len(rows) == len(HYPOTHESES)
