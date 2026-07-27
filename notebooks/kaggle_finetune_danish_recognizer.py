"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach the
'museumscat-danish-recognizer-crops' private dataset (from
src/package_reco_dataset.py) as input, and under Settings enable
Accelerator = GPU T4 x2 (or P100) and Internet = On.

Two-stage recipe (see docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md
"Fine-tune result" section for why): the first frozen-backbone-only attempt
(20 epochs, 240 gradient steps) left the reinitialized vocab-danish
classifier head undertrained -- every date prediction came back MISSING and
recognized text was garbled noise. This recipe gives the head far more
steps to actually learn (Stage 1), then unfreezes the whole network at a
10x lower LR so the conv backbone can adapt slightly to this handwriting
without wrecking its pretrained features (Stage 2). Both stages use
early stopping so they don't run longer than needed.
"""

# %% Cell 1: install doctr and fetch the pinned vendored training script
# !pip install -q "python-doctr[torch]==1.0.1" matplotlib
# !mkdir -p doctr_recognition
# !curl -s -o doctr_recognition/train.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
# !curl -s -o doctr_recognition/utils.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"

# %% Cell 2: locate the attached dataset's train/val folders
# import os
# INPUT_DIR = "/kaggle/input/datasets/matiaspoullain/museumscat-danish-recognizer-crops"
# TRAIN_PATH = os.path.join(INPUT_DIR, "train")
# VAL_PATH = os.path.join(INPUT_DIR, "val")
# assert os.path.isfile(os.path.join(TRAIN_PATH, "labels.json"))
# assert os.path.isfile(os.path.join(VAL_PATH, "labels.json"))

# %% Cell 3: Stage 1 -- frozen backbone, train the reinitialized head much longer
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --pretrained --freeze-backbone \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 150 --batch_size 32 --lr 0.001 \
#     --early-stop --early-stop-epochs 15 --early-stop-delta 0.001 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage1

# %% Cell 4: confirm Stage 1's checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage1.pt")
# print("stage 1 checkpoint ready: /kaggle/working/danish_crnn_vgg16_bn_stage1.pt")

# %% Cell 5: Stage 2 -- unfreeze the whole network, fine-tune at a low LR from Stage 1's weights
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --resume /kaggle/working/danish_crnn_vgg16_bn_stage1.pt \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 50 --batch_size 32 --lr 0.0001 \
#     --early-stop --early-stop-epochs 10 --early-stop-delta 0.001 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage2

# %% Cell 6: confirm the final checkpoint was written -- download this file
# (Kaggle Notebook output pane -> /kaggle/working/danish_crnn_vgg16_bn_stage2.pt)
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
# print("final checkpoint ready to download: /kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
