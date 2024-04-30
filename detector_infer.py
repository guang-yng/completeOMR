import os
import torch, torchvision
import scipy
import numpy as np
from mung.io import read_nodes_from_file, write_nodes_to_file, Node
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
from model import YOLOSoft
from argparse import ArgumentParser
from ultralytics.utils.plotting import Colors
from utils.constants import get_classlist_and_classdict

PATCH_SIZE=1216
MARGIN=128
STEP_SIZE=PATCH_SIZE-2*MARGIN

def bbox_iop(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps

    # Intersection area
    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * max(
        min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0
    )

    return inter / (w1*h1)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='outputs/detection_v8l_b8_i640_essn_s314/train/weights/best.pt', help='The model to load.')
    parser.add_argument('--data', default='data/MUSCIMA++/v2.0', help='The dataset path. Used to link to original images and read ground truths.')
    parser.add_argument('--images', default='data/MUSCIMA++/datasets_r_staff/images', help='The path to images to be predicted.')
    parser.add_argument('-c', '--classes', default='essential',
                        help="The classes used for prediction. Possible values are ['essn', 'essential', '20', 'all']. Default to 'essential'.")
    parser.add_argument('--visualize', action='store_true', help='Whether visualize the result. (draw bounding boxes)')
    parser.add_argument('--grids', action='store_true', help='Whether to visualize the girds. Only valid when --visualize is set.')
    parser.add_argument('--links', action='store_true', help='Whether to generate psuedo edges in annotations.')
    parser.add_argument('--batch_size', default=16, help='The batch size for inference.')
    parser.add_argument('--save_dir', default='data/MUSCIMA++/v2.0_gen_essn_s314', help='The directory to save results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'data'), exist_ok=True)
    save_img_dir = os.path.join(args.save_dir, 'data', 'images')
    if not os.path.exists(save_img_dir):
        os.symlink(os.path.abspath(args.images), save_img_dir, target_is_directory=True)

    print("Loading model...")
    model = YOLOSoft(args.model)
    print("DONE.")
    images = []
    indices = []
    for img_name in tqdm(os.listdir(args.images), desc='splitting images..'):
        img = Image.open(os.path.join(args.images, img_name))
        W, H = img.size
        x_steps, y_steps = (W-1)//STEP_SIZE + 1, (H-1)//STEP_SIZE + 1
        for x_id in range(x_steps):
            for y_id in range(y_steps):
                offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
                subimage = img.crop((offset[0], offset[1], offset[0]+PATCH_SIZE, offset[1]+PATCH_SIZE))
                images.append(subimage)
                indices.append((img_name, x_id, y_id))
    results = []
    for idx in tqdm(range(0, len(images), args.batch_size), desc='predicting...'):
        batch_imgs = images[idx:min(idx+args.batch_size, len(images))]
        results.extend(model(batch_imgs, verbose=False))
    imgname2preds = {}
    for index, result in tqdm(zip(indices, results), desc="merging results...", total=len(indices)):
        img_name, x_id, y_id = index
        offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
        if img_name not in imgname2preds:
            imgname2preds[img_name] = {}
        preds_dict = imgname2preds[img_name]
        preds_list = []
        preds_dict[(x_id, y_id)] = preds_list

        def remove_truncated(plist, true_box, prob):
            to_remove = None
            for idx, (box_pre, prob_pre) in enumerate(plist):
                if bbox_iop(box_pre, true_box)*np.dot(prob.data, prob_pre.data) > 0.20:
                    to_remove = idx
                    break
            if to_remove:
                del plist[to_remove]

        for box, prob in zip(result.boxes, result.probs):
            bbox = box.xyxy.squeeze().cpu().numpy()
            probdata = prob.cpu().numpy()
            left, top, right, bottom = bbox
            if (right <= 2*MARGIN and x_id > 0) or (bottom <= 2*MARGIN and y_id > 0):
                continue
            true_box = bbox + np.array([offset[0], offset[1], offset[0], offset[1]])
            if (x_id > 0 and left < 2*MARGIN):
                assert (x_id-1, y_id) in preds_dict
                remove_truncated(preds_dict[(x_id-1, y_id)], true_box, probdata)
            if (y_id > 0 and top < 2*MARGIN):
                assert (x_id, y_id-1) in preds_dict
                remove_truncated(preds_dict[(x_id, y_id-1)], true_box, probdata)
            preds_list.append((true_box, probdata))
            
    # Visualize
    if args.visualize:
        os.makedirs(os.path.join(args.save_dir, 'data', 'visualization'), exist_ok=True)
        color_palette = Colors()
        for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving visualization files..."):
            img = Image.open(os.path.join(args.images, img_name)).convert("RGB")
            draw = ImageDraw.ImageDraw(img)
            for (x_id, y_id), preds_list in preds_dict.items():
                for box, prob in preds_list:
                    draw.rectangle(tuple(box), outline=color_palette(int(prob.top1)), width=3)
                if args.grids:
                    draw.rectangle((x_id*STEP_SIZE, y_id*STEP_SIZE, (x_id+1)*STEP_SIZE, (y_id+1)*STEP_SIZE), outline=(0, 255, 255), width=8)
                    if x_id == 1 and y_id == 1:
                        draw.rectangle(((x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN), ((x_id+1)*STEP_SIZE+MARGIN, (y_id+1)*STEP_SIZE+MARGIN)), outline=(255, 255, 0), width=6)
            img.save(os.path.join(args.save_dir, 'data', 'visualization', img_name))
    
    # Load classes
    class_list, class_dict = get_classlist_and_classdict(args.classes)
    id2clsname = {idx:clsname for clsname, idx in class_dict.items()}

    # Generate pseudo links
    if args.links:
        imgname2links = {}
        for img_name, preds_dict in tqdm(imgname2preds.items(), desc="creating pseudo links..."):
            pseudo_links = []
            imgname2links[img_name] = pseudo_links
            boxes, probs = [], []
            for preds_list in preds_dict.values():
                for box, prob in preds_list:
                    boxes.append(box)
                    probs.append(prob.data)
                    pseudo_links.append([])
            boxes = np.stack(boxes)
            probs = np.stack(probs)
            doc_name = img_name.split('.')[0]
            nodes = read_nodes_from_file(os.path.join(args.data, 'data', 'annotations', f"{doc_name}.xml"))
            nodes = [node for node in nodes if node.class_name in class_list]
            boxes_g = []
            prob_matrix = []
            id2idx = {}
            for idx, node in enumerate(nodes):
                id2idx[node.id] = idx
                prob_matrix.append(probs[:, class_dict[node.class_name]])
                boxes_g.append((node.left, node.top, node.right, node.bottom))
            prob_matrix = np.stack(prob_matrix).transpose()
            box_matrix = torchvision.ops.box_iou(torch.tensor(boxes), torch.tensor(boxes_g)).numpy()
            cost_matrix = np.multiply(box_matrix, prob_matrix)
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(-cost_matrix)
            match_b = {col_idx:row_idx for row_idx, col_idx in zip(row_indices, col_indices) if cost_matrix[row_idx, col_idx] > 0.05}
            for col_idx in match_b:
                for to_id in nodes[col_idx].outlinks:
                    if to_id not in id2idx:
                        continue
                    if id2idx[to_id] in match_b:
                        pseudo_links[match_b[col_idx]].append(match_b[id2idx[to_id]])

    # Saving results
    os.makedirs(os.path.join(args.save_dir, 'data', 'annotations'), exist_ok=True)
    for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving predictions..."):
        doc_name = img_name.split('.')[0].replace('symbol', 'ideal')
        soft_classes = []
        nodes = []
        for preds_list in preds_dict.values():
            for box, prob in preds_list:
                idx = len(nodes)
                nodes.append(Node(idx, class_name=id2clsname[int(prob.top1)], 
                            top=round(box[1].item()), left=round(box[0].item()), 
                            height=round((box[3]-box[1]).item()), width=round((box[2]-box[0]).item()), 
                            inlinks=[], outlinks=imgname2links[img_name][idx] if args.links else []))
                soft_classes.append(prob.data)
        write_nodes_to_file(nodes, os.path.join(args.save_dir, 'data', 'annotations', f"{doc_name}.xml"), doc_name, f"MUSCIMA-pp_2.0_gen_{args.classes}")
        np.save(os.path.join(args.save_dir, 'data', 'annotations', f"{doc_name}.npy"), soft_classes)
    print("DONE.")
