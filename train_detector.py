import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help="The yaml config file for dataset.")
    parser.add_argument('-e', '--epochs', default=500, type=int, help="Number of training epochs.")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size.")
    parser.add_argument('-m', '--model', default='weights/yolov8l.pt', type=str, help="The model weights to use. Set to the stored weight for validation.")
    parser.add_argument('-i', '--imgsz', default=640, type=int, help="The image size of YOLO. Choose from [640, 1280, 1920].")
    parser.add_argument('--seed', default=0, type=int, help="The random seed.")
    parser.add_argument('--output_dir', default='outputs/detection_v8l_b8_i640', help='The directory to save training info & models.')
    parser.add_argument('--val_only', action='store_true', help='Validate the model only')

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    if not args.val_only:
        model.train(data=args.config, epochs=args.epochs, batch=args.batch_size, imgsz=640, project=args.output_dir, seed=args.seed)  # image size chocie : 640, 1280, and 1920.

    nt_per_class = None
    def save_nt_per_class(validator):
        global nt_per_class
        nt_per_class = validator.nt_per_class

    model.callbacks["on_val_end"].append(save_nt_per_class)      
    metrics = model.val(project=args.output_dir, name="valid") # evaluate model performance on the validation set

    df = pd.DataFrame.from_dict(
        {
            "name": [metrics.names[c] for c in metrics.ap_class_index],
            "count": [nt_per_class[c] for c in metrics.ap_class_index],
            "mAP50": [metrics.class_result(i)[2] for i in range(len(metrics.ap_class_index))]
        }
    )
    df.to_csv(os.path.join(args.output_dir, "result_mAP.csv"))
    print(f"mAP50: {np.mean(df['mAP50'])}")
    print(f"Weighted mAP50: {np.dot(df['count'], df['mAP50'])  / np.sum(df['count'])}")