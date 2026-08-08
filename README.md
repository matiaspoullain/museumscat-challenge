# museumscat-challenge

Kaggle competition: https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task
(MuseumSCAT: Specimen Collection Annotation Task@CVNH ECCV26, hosted by the Natural History Museum of Denmark)

Transcribes `verbatimDate` and `verbatimLocality` from photos of pinned Danish Dung
Beetle specimen labels, with calibrated per-field confidence, scored by AURC over
normalized edit distance. Current approach: a Gemini VLM per-card transcription
pipeline (see `CLAUDE.md`'s "Status" section for the full picture, current numbers,
and proposed next steps).

## Setup

```bash
conda env create -f environment.yml
conda run -n museumscat-challenge python src/download_kaggle_data.py  # data/, models/ are gitignored
```

Gemini pipeline runs need a free-tier `GEMINI_API_KEY` in the environment (Google AI
Studio, not Vertex AI).

## Running tests

```bash
conda run -n museumscat-challenge python -m pytest
```

## Docs

- `CLAUDE.md` — current status, environment, and competition-specific constraints.
- `kaggle_meta/pages/` — competition rules/description, fetched verbatim.
- `docs/superpowers/plans/` and `docs/superpowers/specs/` — implementation plans and
  design docs, in chronological order (see `CLAUDE.md` for which ones are current).
