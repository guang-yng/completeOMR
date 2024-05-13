# Toward a more complete OMR solution
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.11-yellow)

This codebase will be cleaned, simplified, and made public after review.

## Data Preparation
The MUSCIMA++ v2.0 dataset is publicly available at [here](https://github.com/OMR-Research/muscima-pp/releases/tag/v2.0). 
Download and unzip the dataset with the following commands.
```bash
wget -O data/MUSCIMA++.zip https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip
unzip data/MUSCIMA++.zip -d data/MUSCIMA++
```

Then, we need to download staff-line removed images from CVC-MUSCIMA. To make this simple, you can download the zip file from [here](https://cswashingtonedu-my.sharepoint.com/:u:/g/personal/gyang1_cs_washington_edu/ERqGYflRtJdNqam7zO2B24gBzjcY2CkfFb5inyHv2KckTQ?e=N2eAzE), and move the zip file to the `data/` directory.

Simply run the following command to unzip:
```bash
unzip -d data/MUSCIMA++ data/datasets_r_staff.zip
```

After unzipping the datasets, we expect the directories organized like this:
```
data/
└── MUSCIMA++
    ├── datasets_r_staff
    │   └── images
    └── v2.0
        ├── data
        │   └── annotations
        └── specifications
```


## Environment 

All of our experiments are run with python 3.10.13.

To set up the environment, please run:
```bash
pip install -r requirements.txt
```

## Object Detection

### Data Generation

Use the scrip `utils/detector_data.py` to generate a object detection dataset.

Here are some examples below:

Generate a randomly cropped dataset with essential classes:
```bash
python utils/detector_data.py 
```

Generate a randomly cropped dataset with 20 classes only:
```bash
python utils/detector_data.py --classes 20 --save_dir data/MUSCIMA++/datasets_r_staff_20_crop --save_config data_staff_removed_20_crop.yaml 
```

Generate a randomly cropped dataset with all classes:
```bash
python utils/detector_data.py --classes all --save_dir data/MUSCIMA++/datasets_r_staff_all_crop --save_config data_staff_removed_all_crop.yaml
```

### Training

Run `train_detector.py` to train a object detection model.

Here is an example:
```bash
python train_detector.py --config data/MUSCIMA++/datasets_r_staff_essn_crop/data_staff_removed_crop.yaml
```

### Inference

After training a model, to detect all objects in the whole dataset,
run the following script:

```bash
python detector_infer.py --model outputs/detection_v8l_b8_i640/train/weights/best.pt --save_dir data/MUSCIMA++/v2.0_gen --visualize --grids --links
```

## Notation Assembly

### Training

After training the object detector and generating all of the bounding boxes,
we can simply train our notation assembly model with the script `train_assembler.py`.

Here is an example:
```bash
python train_assembler.py -m data/MUSCIMA++/v2.0_gen/data \
        --model_config configs/assembler/on_gen/MLP32_v2gen.yaml \
        --exp_name assembly_on_gen
```

### Test

To test a trained model on the test set, run following command:
```bash
python train_assembler.py -m data/MUSCIMA++/v2.0_gen/data \
        --model_config configs/assembler/on_gen/MLP32_v2gen.yaml \
        --exp_name assembly_on_gen  --test_only --load_epochs 180 
```