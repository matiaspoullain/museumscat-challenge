"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach two input datasets --
'museumscat-danish-recognizer-word-boxes' (from
src/package_reco_dataset_v2.py, Task 7) and
'museum-card-crops-train-test' (the card crops + manifest.csv dataset,
already uploaded before this plan) -- and under Settings enable
Accelerator = GPU T4 x2 (or P100) and Internet = On.

Two-stage recipe (frozen-backbone warm-up, then full unfreeze at low LR)
reused from the earlier, now-superseded word-crop fine-tune attempt, which
converged with this exact shape -- see
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md
for why that attempt's *dataset* is being replaced, not this recipe.
"""

# %% Cell 1: install doctr and fetch the pinned vendored training script
# !pip install -q "python-doctr[torch]==1.0.1"
# !mkdir -p doctr_recognition
# !curl -s -o doctr_recognition/train.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
# !curl -s -o doctr_recognition/utils.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"

# %% Cell 2: locate the attached word-box dataset's train/val folders
# import os
# INPUT_DIR = "/kaggle/input/museumscat-danish-recognizer-word-boxes"
# TRAIN_PATH = os.path.join(INPUT_DIR, "train")
# VAL_PATH = os.path.join(INPUT_DIR, "val")
# assert os.path.isfile(os.path.join(TRAIN_PATH, "labels.json"))
# assert os.path.isfile(os.path.join(VAL_PATH, "labels.json"))

# %% Cell 3: Stage 1 -- frozen backbone, train the reinitialized head
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --pretrained --freeze-backbone --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 1500 --batch_size 16 --lr 0.001 \
#     --early-stop --early-stop-epochs 120 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage1

# %% Cell 4: confirm Stage 1's checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage1.pt")
# print("stage 1 checkpoint ready")

# %% Cell 5: Stage 2 -- unfreeze the whole network, fine-tune at a low LR from Stage 1's weights
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --resume /kaggle/working/danish_crnn_vgg16_bn_stage1.pt --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 800 --batch_size 16 --lr 0.0001 --optim adamw --wd 0.0001 \
#     --early-stop --early-stop-epochs 100 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage2

# %% Cell 6: confirm the final checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
# print("final checkpoint ready")

# %% Cell 7: locate the already-uploaded card-crops dataset (images + manifest)
# import os
# CARDS_INPUT_DIR = "/kaggle/input/museum-card-crops-train-test"
# CARDS_IMAGES_DIR = os.path.join(CARDS_INPUT_DIR, "images")
# MANIFEST_PATH = os.path.join(CARDS_INPUT_DIR, "manifest.csv")
# assert os.path.isfile(MANIFEST_PATH)

# %% Cell 8: run the fine-tuned recognizer over every card crop, one row per recognized word
# import cv2
# import pandas as pd
# import torch
# from doctr.datasets import VOCABS
# from doctr.models import ocr_predictor, recognition
#
# reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
# reco_model.load_state_dict(
#     torch.load("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt", map_location="cpu", weights_only=True)
# )
# predictor = ocr_predictor(det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True)
#
# manifest = pd.read_csv(MANIFEST_PATH)
# rows = []
# for crop_filename in manifest["crop_filename"]:
#     img_bgr = cv2.imread(os.path.join(CARDS_IMAGES_DIR, crop_filename))
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     result = predictor([img_rgb])
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     rows.append({
#                         "crop_filename": crop_filename,
#                         "predicted_text": word.value,
#                         "confidence": word.confidence,
#                     })
# pd.DataFrame(rows, columns=["crop_filename", "predicted_text", "confidence"]).to_csv(
#     "/kaggle/working/predictions.csv", index=False
# )
# print(f"wrote {len(rows)} word rows for {manifest['crop_filename'].nunique()} cards")

# %% Cell 9: confirm predictions.csv was written
# import os
# assert os.path.isfile("/kaggle/working/predictions.csv")
# print("predictions ready: /kaggle/working/predictions.csv")
