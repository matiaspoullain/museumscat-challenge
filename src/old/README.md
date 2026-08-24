# `src/old` — codigo fuera de la ruta activa

Movido aca el 2026-08-24. **Nada de esto se importa desde el pipeline que
produce el mejor submission** (`src.generate_submission_gemini`, ruta
whole-image-primary, AURC publico 0.02382). Se conserva, no se borra, porque:

- documenta como se generaron artefactos que *si* se siguen usando
  (`data/processed/card_crops/`, el dataset del reconocedor danes), y
- varios modulos son la medicion de un callejon sin salida ya cerrado, que es
  justamente lo que evita que se vuelva a proponer.

Los tests siguen en `tests/` y siguen corriendo: importan `src.old.<modulo>`.

## Que hay aca, por grupo

### 1. Pipeline OCR clasico (superado, AURC 0.576)

Reemplazado por la transcripcion VLM con Gemini (ver CLAUDE.md, Status
2026-08-07 en adelante).

| modulo | que hacia |
|---|---|
| `benchmark_ocr.py` | comparaba motores OCR sobre recortes ROI reales |
| `score_ocr_dev.py` | puntuaba una eleccion de motor/reconocedor en dev (NED + AURC) |
| `score_hybrid_dev.py` | lo mismo para la estrategia hibrida doctr+easyocr |
| `evaluate_card_reco.py` | evaluaba el reconocedor danes fine-tuneado |
| `evaluate_local.py` | evaluacion end-to-end del pipeline clasico en las 200 filas |
| `predict_cards.py` | aplanaba la salida de doctr a filas palabra-a-palabra |

### 2. Entrenamiento del reconocedor danes (fine-tune, ya no se usa)

| modulo | que hacia |
|---|---|
| `extract_reco_crops.py` | extraia recortes de palabra de las imagenes de dev |
| `build_word_crop_dataset.py` | armaba el set imagen+etiqueta a nivel palabra |
| `split_reco_dataset.py`, `split_reco_dataset_v2.py` | split train/val |
| `package_reco_dataset.py`, `package_reco_dataset_v2.py` | empaquetado para subirlo como dataset privado de Kaggle |

### 3. EDA e inspeccion visual (scripts de un solo uso)

`eda_card_blur_variance_test.py`, `eda_card_variance_test.py`,
`eda_label_layout.py`, `eda_sam_card_detection.py`, `inspect_detection.py`,
`review_detections.py`, `review_detections_check.py`.

Calibraron las constantes de `src.detect_labels` / `src.detect_cards`, que si
siguen vivos.

### 4. Apps locales de anotacion manual (su salida ya esta capturada)

`box_card_words_app.py`, `label_crops_app.py`, `match_card_labels_app.py`.

### 5. Analisis y experimentos de confianza ya cerrados

| modulo | resultado |
|---|---|
| `analyze_gemini_errors.py`, `analyze_confidence_errors.py` | mineria de patrones de error; alimentaron `src.clean_gemini_predictions` y `src.confidence_penalty_rules` |
| `analyze_rank_headroom.py` | midio el techo del oraculo de ranking (dev 0.0077) |
| `analyze_local_vlm_sample_count.py` | mostro que la señal de acuerdo del VLM local ya esta presente con `n_samples=1` |
| `experiment_confidence_grid.py`, `gemini_calibration_features.py` | grilla de calibradores 1-D y multi-feature: **ninguno le gano a la confianza cruda** |
| `consensus_text.py` | Track B (seleccion MBR, snap al corpus): **medido como dañino / no adoptado** |
