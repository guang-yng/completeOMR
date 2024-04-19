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
python utils/detector_data.py --classes 20 --save_dir MUSCIMA++/datasets_r_staff_20_crop --save_config data_staff_removed_20_crop.yaml 
```

Generate a randomly cropped dataset with all classes:
```bash
python utils/detector_data.py --classes all --save_dir MUSCIMA++/datasets_r_staff_all_crop --save_config data_staff_removed_all_crop.yaml
```

### Training

Run `train_detector.py` to train a object detection model.

Here is an example:
```bash
python train_detector.py --config data/MUSCIMA++/datasets_r_staff_essn_crop/data_staff_removed_crop.yaml
```

<!---
### Inference

To obtain the predictions on test dataset, run the following script:
```
usage: infer.py [-h] [--model MODEL] [--data DATA] [--images IMAGES] [--classes CLASSES] [--visualize] [--grids] [--links]
                [--batch_size BATCH_SIZE] [--save_dir SAVE_DIR]

options:
  -h, --help            show this help message and exit
  --model MODEL         The model to load.
  --data DATA           The dataset path. Used to link to original images and read ground truths.
  --images IMAGES       The path to images to be predicted.
  --classes CLASSES     The classes used for inference. Possible values are ['essential', '20', 'all']. Default to 'essential'.
  --visualize           Whether visualize the result. (draw bounding boxes)
  --grids               Whether to visualize the girds. Only valid when --visualize is set.
  --links               Whether to generate psuedo edges in annotations.
  --batch_size BATCH_SIZE
                        The batch size for inference.
  --save_dir SAVE_DIR   The directory to save results
```

For example, to run inference on essential classes and visualize the result:
```bash
python infer.py --visualize --links
```

To show segmetation grids in the visualization, add `--grids` option.

We also provide the generated detection output in `./data/v2.0_gen_essn.zip`.
Run the following command to extract them:
```bash
unzip -d data/MUSCIMA++/ data/v2.0_gen_essn.zip
```


## Notation Assembly

After training the object detector and generating all of the bounding boxes,
we can simply train our notation assembly model using following commands (All of the experiments will run with 3 different seeds):

For baseline MLP, 
```bash
bash run_vanilla.sh
```

For baseline MLP + pipeline training,
```bash
bash run_v2gen_v0.sh
```

For baseline MLP + pipeline training + soft label,
```bash
bash run_v2gen_v0_soft.sh
```

All of the training are done with essential classes,
please check these scripts for possible options.

To test a trained model on the test set, run following command:
```
python -m effnet.train -m data/MUSCIMA++/v2.0_gen_essn/data \
        --model_config configs/effnet/MLP32_nogrammar.yaml \
        --exp_name vanilla_reproduce_$seed --test_only --edgelist_inf --load_epochs $best_model_epoch
```
--->