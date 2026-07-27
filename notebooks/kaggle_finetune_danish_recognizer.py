"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach the
'museumscat-danish-recognizer-crops' private dataset (from
src/package_reco_dataset.py) as input, and under Settings enable
Accelerator = GPU T4 x2 (or P100) and Internet = On.
"""

# %% Cell 1: install doctr and fetch the pinned vendored training script
# !pip install -q "python-doctr[torch]==1.0.1" matplotlib
# !mkdir -p doctr_recognition
# !curl -s -o doctr_recognition/train.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
# !curl -s -o doctr_recognition/utils.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"

# %% Cell 2: locate the attached dataset's train/val folders
# import os
# INPUT_DIR = "/kaggle/input/museumscat-danish-recognizer-crops"
# TRAIN_PATH = os.path.join(INPUT_DIR, "train")
# VAL_PATH = os.path.join(INPUT_DIR, "val")
# assert os.path.isfile(os.path.join(TRAIN_PATH, "labels.json"))
# assert os.path.isfile(os.path.join(VAL_PATH, "labels.json"))

# %% Cell 3: run the fine-tune (freeze backbone, extend vocab to Danish)
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --pretrained --freeze-backbone \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 20 --batch_size 32 --lr 0.001 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn

# %% Cell 4: confirm the checkpoint was written -- download this file
# (Kaggle Notebook output pane -> /kaggle/working/danish_crnn_vgg16_bn.pt)
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn.pt")
# print("checkpoint ready to download: /kaggle/working/danish_crnn_vgg16_bn.pt")
