"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach the
'museumscat-danish-recognizer-crops' private dataset (from
src/package_reco_dataset.py) as input, and under Settings enable
Accelerator = GPU T4 x2 (or P100) and Internet = On.

Two-stage recipe (see docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md
"Fine-tune result" section for the full history): the first frozen-backbone
attempt (20 epochs, 240 gradient steps) left the reinitialized vocab-danish
classifier head undertrained -- every date prediction came back MISSING.
A first Stage 1/2 attempt (150/50 epoch ceilings, patience 15/10) fixed
that but converged to a mixed result (locality NED beat baseline, date NED
did not) and finished in about a minute -- meaning early stopping tripped
almost immediately, likely on validation-loss noise from only ~4-7 batches
per epoch, not real convergence.

This version pushes both duration and training complexity, budgeted for
roughly an hour of GPU time:
  - Higher epoch ceilings (600 / 400) so there's room to actually converge.
  - Much more patient early stopping (60 / 50 epochs) so a noisy small
    validation set doesn't trigger a premature stop.
  - Smaller batch size (16, was 32) -- doubles gradient updates per epoch
    on this small (391-crop) dataset.
  - Stage 2 (the riskier full-backbone unfreeze, on the same 391 crops)
    switches to AdamW + weight decay for extra regularization.
  - --amp (mixed precision) on both stages to keep this within the time
    budget on a T4's tensor cores.
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
#     --vocab danish --pretrained --freeze-backbone --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 600 --batch_size 16 --lr 0.001 \
#     --early-stop --early-stop-epochs 60 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage1

# %% Cell 4: confirm Stage 1's checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage1.pt")
# print("stage 1 checkpoint ready: /kaggle/working/danish_crnn_vgg16_bn_stage1.pt")

# %% Cell 5: Stage 2 -- unfreeze the whole network, fine-tune at a low LR from Stage 1's weights
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --resume /kaggle/working/danish_crnn_vgg16_bn_stage1.pt --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 400 --batch_size 16 --lr 0.0001 --optim adamw --wd 0.0001 \
#     --early-stop --early-stop-epochs 50 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage2

# %% Cell 6: confirm the final checkpoint was written -- download this file
# (Kaggle Notebook output pane -> /kaggle/working/danish_crnn_vgg16_bn_stage2.pt)
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
# print("final checkpoint ready to download: /kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
