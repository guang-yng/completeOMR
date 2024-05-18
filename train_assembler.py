import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.data_pool import load_munglinker_data
from mung.io import read_nodes_from_file

from utils.constants import get_classlist_and_classdict
from utils.metrics import compute_matching_score
from utils.utility import set_seed
from configs.assembler.default import get_cfg_defaults
from model import MLP, MLPwithSoftClass, MLPwithSoftClassExtraMLP

import argparse
import os
import tqdm
import glob
import yaml

def build_model(cfg):
    if cfg.MODEL.MODE == "MLP":
        model = MLP(cfg)
    elif cfg.MODEL.MODE == "MLPwithSoftClass":
        model = MLPwithSoftClass(cfg)
    elif cfg.MODEL.MODE == "MLPwithSoftClassExtraMLP":
        model = MLPwithSoftClassExtraMLP(cfg)
    else:
        raise ValueError(f"Model {cfg.TRAIN.MODEL} is not supported")

    return model

def save_checkpoint(epoch, model, optimizer, save_file):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)

def load_checkpoint(model, optimizer, load_file):
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded model at epoch {checkpoint['epoch']}.")


def train(args, data, cfg, device, model, class_list, class_dict):
    with open(f"{args.output_dir}/{args.exp_name}/config.yaml", 'w') as f:
        f.write(cfg.dump())
        
    if cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    else:
        raise ValueError(f"Optimizer {cfg.TRAIN.OPTIMIZER} is not supported")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT))
    loader = DataLoader(data['train'], batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    if args.load_epochs > 0:
        load_checkpoint(model, optimizer, f"{args.output_dir}/{args.exp_name}/model_ep{args.load_epochs}.pth")
        for _ in tqdm.tqdm(range(args.load_epochs), desc="Skipping Epochs"):
            torch.empty((), dtype=torch.int64).random_()
            torch.empty((), dtype=torch.int64).random_()

    model.train()
    for epoch in range(args.load_epochs, cfg.TRAIN.NUM_EPOCHS):
        corr = 0
        total = 0
        for batch in tqdm.tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['label'])
            loss.backward()
            optimizer.step()

            pred = torch.sigmoid(output) > 0.5
            corr += (pred == batch['label']).sum().item()
            total += len(batch['label'])
        print(f"Epoch {epoch+1} accuracy: {corr/total}")

        if (epoch+1) % cfg.TRAIN.EVAL_FREQUENCY == 0:
            print(f"Validating Epoch {epoch+1}")
            with torch.no_grad():
                evaluate(args, data, cfg, device, model, class_list, class_dict)
        if (epoch+1) % cfg.TRAIN.SAVE_FREQUENCY == 0 and (epoch+1) != cfg.TRAIN.NUM_EPOCHS:
            save_checkpoint(epoch+1, model, optimizer, f"{args.output_dir}/{args.exp_name}/model_ep{epoch+1}.pth")

    save_checkpoint(cfg.TRAIN.NUM_EPOCHS, model, optimizer, f"{args.output_dir}/{args.exp_name}/model_final.pth")

    return model

def evaluate(args, data, cfg, device, model, class_list, class_dict):
    model.eval()
    all_mung_files = glob.glob(args.mung_root + "/**/*.xml", recursive=True)
    all_gt_files = glob.glob(args.gt_mung_root + "/**/*.xml", recursive=True)
    with open(args.split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)
    if args.test_only:
        include_names = split['test']
        data = data['test']
    else:
        include_names = split['valid']
        data = data['valid']
    mung_files_in_this_split = sorted([f for f in all_mung_files if os.path.splitext(os.path.basename(f))[0] in include_names])
    gt_files_in_this_split = sorted([f for f in all_gt_files if os.path.splitext(os.path.basename(f))[0] in include_names])
    class_prob_files = glob.glob(args.mung_root + "/**/*.npy", recursive=True)
    class_prob_files = sorted([f for f in class_prob_files if os.path.splitext(os.path.basename(f))[0] in include_names])
    USE_HARD_LABEL = len(class_prob_files) == 0
    if USE_HARD_LABEL:
        print("No soft label found. Use top-1 hard labels instead.")

    inference_graph = data.get_inference_graph() if isinstance(data.inference_graph[0], list) else data.inference_graph
    total_matching_score = np.array([0.0, 0.0, 0.0, 0.0])
    for i in tqdm.tqdm(range(len(mung_files_in_this_split)), desc="inferencing..."):
        mung_file = mung_files_in_this_split[i]
        gt_file = gt_files_in_this_split[i]
        node_list = read_nodes_from_file(mung_file)
        gt_list = read_nodes_from_file(gt_file)
        if USE_HARD_LABEL:
            class_label = np.array([class_dict[node.class_name] for node in node_list])
            class_prob = np.zeros((len(node_list), cfg.MODEL.VOCAB_DIM))
            class_prob[np.arange(len(class_label)), class_label] = 1
        else:
            class_prob =  np.load(class_prob_files[i])
        edge_list = []
        
        cur_graph = inference_graph[i]
        for batch_idx in range((cur_graph['source_id'].shape[0] // cfg.EVAL.BATCH_SIZE)+1):
            batch = {k: v[batch_idx*cfg.EVAL.BATCH_SIZE : batch_idx*cfg.EVAL.BATCH_SIZE + cfg.EVAL.BATCH_SIZE].to(device) 
                     for k, v in cur_graph.items()}
            output = model(batch)
            output = torch.sigmoid(output)
            for idx in range(batch['source_id'].shape[0]):
                source_id = batch['source_id'][idx]
                target_id = batch['target_id'][idx]
                edge_list.append((source_id.item(), target_id.item(), output[idx].item()))
        matching_score = compute_matching_score(node_list, class_prob, edge_list, gt_list, class_list, class_dict)
        total_matching_score += matching_score

    print(f"Average (AUC, F1, Precision, Recall) score: {(total_matching_score / len(mung_files_in_this_split)).tolist()}")


def main(args, data, cfg, device, class_list, class_dict):

    model = build_model(cfg)
    model.to(device)
    
    if args.test_only or args.val_only:
        # Load model for test or validation
        if args.load_epochs == -1: # test/val last checkpoint
            if os.path.exists(f"{args.output_dir}/{args.exp_name}/model_final.pth"):
                all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_final.pth"]
            else:
                model_files = [f for f in os.listdir(f"{args.output_dir}/{args.exp_name}") if f.startswith("model_ep") and f.endswith(".pth")]
                assert len(model_files) > 0, "No checkpoint detected!"
                model_files = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0][2:]), reverse=True)
                all_ckpts = [f"{args.output_dir}/{args.exp_name}/{model_files[0]}"]
        elif args.load_epochs == -2: # test/val all checkpoints
            all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_ep{e}.pth" for e in range(cfg.TRAIN.SAVE_FREQUENCY, cfg.TRAIN.NUM_EPOCHS, cfg.TRAIN.SAVE_FREQUENCY)] + [f"{args.output_dir}/{args.exp_name}/model_final.pth"]
        else: # test/val specified checkpoints
            all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_ep{args.load_epochs}.pth"]

        for ckpt in all_ckpts:
            load_checkpoint(model, None, ckpt)
            evaluate(args, data, cfg, device, model, class_list, class_dict)

    else: # For train
        args.load_epochs = 0 if args.load_epochs < 0 else args.load_epochs
        model = train(args, data, cfg, device, model, class_list, class_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mung_root', default="data/MUSCIMA++/v2.0_gen_essn/data/", help="The root directory of the detection output")
    parser.add_argument('-g', '--gt_mung_root', default="data/MUSCIMA++/v2.0/data/", help="The root directory of the ground truth MUSCIMA++ dataset")
    parser.add_argument('-s', '--split_file', default="splits/mob_split.yaml", help="The split file")
    parser.add_argument('-c', '--classes', default='essential',
                    help="The classes used for prediction. Possible values are ['essn', 'essential', '20', 'all']. Default to 'essential'.")
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--val_only', action="store_true")
    parser.add_argument('--test_only', action="store_true")
    parser.add_argument('--load_epochs', type=int, default=-1)
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER, help="options to overwrite the config")

    args = parser.parse_args()

    if args.test_only and args.val_only:
        raise ValueError("test_only and val_only can't be used at the same time.")

    cfg = get_cfg_defaults()
    if args.test_only or args.val_only:
        cfg.merge_from_file(os.path.join(args.output_dir, args.exp_name, "config.yaml"))
    if args.model_config and not args.test_only and not args.val_only:
        cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(args.opts)

    print(f"Experiment name: {args.exp_name}")

    class_list, class_dict = get_classlist_and_classdict(args.classes)

    with open(cfg.DATA.DATA_CONFIG, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE

    if args.test_only:
        data = load_munglinker_data(
            mung_root=args.mung_root,
            images_root=args.mung_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=False,
            load_validation_data=False,
            load_test_data=True,
        )
    elif args.val_only:
        data = load_munglinker_data(
            mung_root=args.mung_root,
            images_root=args.mung_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=False,
            load_validation_data=True,
            load_test_data=False,
        )
    else:
        data = load_munglinker_data(
            mung_root=args.mung_root,
            images_root=args.mung_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=True,
            load_validation_data=True,
            load_test_data=False,
        )

    if cfg.SYSTEM.NUM_GPUS > 0:
        if not torch.cuda.is_available():
            raise ValueError("No GPU available")
        device = "cuda"
    else:
        device = "cpu"
    
    set_seed(cfg.SYSTEM.SEED)
    os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)

    main(args, data, cfg, device, class_list, class_dict)


