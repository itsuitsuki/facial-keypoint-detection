UC Berkeley CS180 FA24 Final Project 1/3: Facial Keypoint Detection using NN

# File Structure
```
.
├── 1
├── 1_imm_nose_tip_detection.ipynb
├── 2
├── 2_full_facial_keyp_detection.ipynb
├── 3
├── 3_ibug_facial.ipynb
├── 4
├── 4_ibug_heatmap.ipynb
├── ibug_300W_large_face_landmark_dataset # downloaded from kaggle and project spec
├── imm_face_db # imm face dataset
├── my_ims # images from my collection
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── README.md
└── unet.py
```
# How to run
1. First, please install `albumentations` and `torchsummary` by running `pip install albumentations torchsummary`.
2. Every notebook is corresponding to a part of the project, and can be run independently.
3. In the first part, I write a method for computing the real nose tips as labels by weighted averages, but because this is not explicitly said to be allowed in the project spec, I just preserve the implementation in the notebook and set `REAL_NOSE_TIP = False` to use the provided labels (52-th index).
4. `unet.py` is the implementation of the U-Net model, which is used in the project. Also the aggregation (weighted average) function `heatmap_to_landmarks` is defined in this file.
5. The iBUG dataset is not included in the repo, but can be downloaded from the project spec or kaggle. IMM Face dataset can also be downloaded through the wayback-ed IMM homepage. The datasets should be placed in the root directory of the project.
6. The `my_ims` directory contains some images from my collection, which can be used to test the model in part 3 and 4.
