"""Tests of the predictions of the broadcast swap of step P3, written before the code.

The rules are in docs/decisions/2026_11_24_p3_swap_predictions.md. Each expected band below comes from the
table "The donor classes" of that document, not from a run.
"""
import pytest

from scripts.analysis.swap_predictions import (
    HYPOTHESES,
    Episode,
    donor_class,
    predicted_band,
    separates,
)

RECIPIENT = Episode(target_band=3, slot_order=(3, 7))
C1_DONOR = Episode(target_band=7, slot_order=(3, 7))
C2_DONOR = Episode(target_band=7, slot_order=(7, 3))
C3_DONOR = Episode(target_band=3, slot_order=(7, 3))
C0_DONOR = Episode(target_band=3, slot_order=(3, 7))

OWN, OTHER = 3, 7
EXPECTED = {
    "C1": {"ignore": OWN, "band": OTHER, "slot": OTHER},
    "C2": {"ignore": OWN, "band": OTHER, "slot": OWN},
    "C3": {"ignore": OWN, "band": OWN, "slot": OTHER},
    "C0": {"ignore": OWN, "band": OWN, "slot": OWN},
}
DONORS = {"C1": C1_DONOR, "C2": C2_DONOR, "C3": C3_DONOR, "C0": C0_DONOR}


def test_hypotheses_are_the_three_of_the_rules():
    assert HYPOTHESES == ("ignore", "band", "slot")


@pytest.mark.parametrize("label", ["C1", "C2", "C3", "C0"])
def test_donor_class_of_each_donor(label):
    assert donor_class(RECIPIENT, DONORS[label]) == label


@pytest.mark.parametrize("label", ["C1", "C2", "C3", "C0"])
@pytest.mark.parametrize("hypothesis", ["ignore", "band", "slot"])
def test_predicted_band_matches_the_table(label, hypothesis):
    assert predicted_band(hypothesis, RECIPIENT, DONORS[label]) == EXPECTED[label][hypothesis]


def test_predictions_do_not_depend_on_band_numbers():
    recipient = Episode(target_band=1, slot_order=(0, 1))
    donor = Episode(target_band=1, slot_order=(1, 0))
    assert donor_class(recipient, donor) == "C3"
    assert predicted_band("slot", recipient, donor) == 0


def test_self_replacement_predicts_the_native_band_for_every_hypothesis():
    assert {predicted_band(h, RECIPIENT, C0_DONOR) for h in HYPOTHESES} == {OWN}


@pytest.mark.parametrize("label", ["C1", "C2", "C3", "C0"])
def test_one_class_alone_does_not_separate(label):
    assert separates([(RECIPIENT, DONORS[label])]) is False


def test_c1_and_c2_reported_apart_separate():
    # Rule 3. Rows over (C1, C2): ignore (own, own), band (other, other), slot (other, own).
    assert separates([(RECIPIENT, C1_DONOR), (RECIPIENT, C2_DONOR)]) is True


def test_c1_and_c3_separate_and_c0_adds_nothing():
    assert separates([(RECIPIENT, C1_DONOR), (RECIPIENT, C3_DONOR)]) is True
    assert separates([(RECIPIENT, C0_DONOR), (RECIPIENT, C0_DONOR)]) is False


def test_all_classes_together_separate():
    assert separates([(RECIPIENT, DONORS[label]) for label in ("C1", "C2", "C3", "C0")]) is True


def test_empty_design_does_not_separate():
    assert separates([]) is False


def test_unknown_hypothesis_raises():
    with pytest.raises(ValueError):
        predicted_band("recombine", RECIPIENT, C1_DONOR)


@pytest.mark.parametrize("bad", [
    dict(target_band=3, slot_order=(3, 3)),
    dict(target_band=5, slot_order=(3, 7)),
    dict(target_band=3, slot_order=(3,)),
])
def test_malformed_episode_raises(bad):
    with pytest.raises(ValueError):
        Episode(**bad)


def test_swap_with_other_bands_is_in_no_class():
    stranger = Episode(target_band=4, slot_order=(4, 9))
    with pytest.raises(ValueError):
        donor_class(RECIPIENT, stranger)
    with pytest.raises(ValueError):
        predicted_band("band", RECIPIENT, stranger)
