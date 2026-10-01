"""Local VLM (Qwen2.5-VL) second-opinion signal, re-purposed as a within-tier
tiebreaker (src.confidence_rank_rules.combine_tier_and_within_tier_score)
instead of the closed spike's blend/gate designs
(docs/superpowers/plans/2026-08-19-local-vlm-second-opinion-spike.md), which
both failed for the same stated reason: "the real information the local VLM
has turns out to concentrate in exactly the rows Gemini is already sure
about" -- i.e. inside the confidence=1.0 tie the spike never actually tested
reordering, since it only tried blending across confidence levels (harmful)
and gating at low confidence (no effect).

Re-analyzed here on the same already-collected 152/197 dev crops
(data/processed/gemini/kaggle_vlm_spike_dataset/previous_card_results.csv):
within Gemini's top confidence tier, specimens where the local VLM AGREES
with Gemini's final text have mean NED 0.000 (verbatimDate, n=21) / 0.039
(verbatimLocality, n=25); specimens where it DISAGREES have 0.033 (n=20) /
0.193 (n=23) -- a real, mechanistically clear gradient, not previously tested.
See docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md for the
full evaluation (combined with the rest of src.confidence_rank_rules: dev
AURC 0.0575 -> 0.0493).

Zero additional Gemini quota -- this only reads a second model's already-run
outputs. build_local_vlm_predictions_by_crop reads the spike's own CSV schema
(local_vlm_verbatimDate, not verbatimDate, to keep it visually distinct from
Gemini's own predictions files); a missing file (the run hasn't happened for
this split yet -- e.g. the test-set extension before it's run) returns {}
rather than raising, same convention as
src.calibrate_gemini_confidence.load_wholeimage_predictions.
"""
import csv
import os

from src.merge_records import FieldScores

FIELDS = ("verbatimDate", "verbatimLocality")

DEV_LOCAL_VLM_CARD_PATH = os.path.join(
    "data", "processed", "gemini", "kaggle_vlm_spike_dataset", "previous_card_results.csv"
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_local_vlm_predictions_by_crop(path: str = DEV_LOCAL_VLM_CARD_PATH) -> dict[str, FieldScores]:
    if not os.path.exists(path):
        return {}
    by_crop: dict[str, FieldScores] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_crop[row["crop_filename"]] = {
                "verbatimDate": (row["local_vlm_verbatimDate"], float(row["local_vlm_verbatimDate_confidence"])),
                "verbatimLocality": (
                    row["local_vlm_verbatimLocality"], float(row["local_vlm_verbatimLocality_confidence"])
                ),
            }
    return by_crop


def covered_crops_by_specimen(manifest_by_image: dict[str, list[str]],
                              local_predictions_by_crop: dict[str, FieldScores]) -> dict[str, list[str]]:
    """image_file -> [crop_filename, ...] restricted to crops the local VLM
    actually has a prediction for -- coverage is partial (152/197 dev crops
    in the pilot run), so callers must be able to tell "not covered" (empty
    list) apart from "covered but found nothing", which
    local_vlm_agreement_score needs to fall back to neutral correctly."""
    return {
        image_file: [c for c in crop_filenames if c in local_predictions_by_crop]
        for image_file, crop_filenames in manifest_by_image.items()
    }


def local_vlm_agreement_score(image_file: str, field: str, gemini_text: str,
                              local_merged: dict[str, FieldScores],
                              covered_crops: dict[str, list[str]]) -> float:
    """[0.0, 0.5, 1.0]: 1.0 if the local VLM's specimen-level merged text for
    `field` (case/whitespace-normalized) matches Gemini's final text, 0.0 if
    it disagrees, 0.5 (neutral -- leaves within_tier_score's other components
    untouched when averaged in) whenever there's nothing to compare: the
    specimen has no local-VLM-covered crop at all, Gemini's own text is
    "MISSING" (nothing to agree or disagree about), or the local VLM's merged
    text for this specimen is itself "MISSING" (it found nothing, which is a
    different claim than "it found something different")."""
    if not covered_crops.get(image_file) or gemini_text == "MISSING":
        return 0.5
    local_text, _ = local_merged.get(image_file, {}).get(field, ("MISSING", 0.5))
    if local_text == "MISSING":
        return 0.5
    return 1.0 if _normalize(local_text) == _normalize(gemini_text) else 0.0
