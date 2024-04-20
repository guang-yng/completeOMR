import os
import yaml
import shutil
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from mung.io import read_nodes_from_file
from constants import get_classlist_and_classdict
from utility import set_seed

def load_split(split_file):
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)
    return split

def generate(docs, class_list, class_dict, save_dir, mode, crop_times, image_source_dir):
    images_dir = os.path.join(save_dir, mode, 'images')
    labels_dir = os.path.join(save_dir, mode, 'labels')
    # Create (or clear) directories
    for directory in [images_dir, labels_dir]:
        os.makedirs(directory, exist_ok=True)  # Creates the directory if it doesn't exist, does nothing otherwise

    for doc in tqdm(docs, desc=f'Generating {mode}'):
        doc_name = doc[0].document
        src_path = os.path.join(image_source_dir, f"{doc_name}.png")
        
        # Open the image to get its dimensions
        with Image.open(src_path) as img:
            img_width, img_height = img.size
            
        if crop_times == 0 or mode == 'test':
            # Copy the image to the target file
            dst_path = os.path.join(images_dir, f"{doc_name}.png")
            shutil.copy(src_path, dst_path)
            
            # Open label file for writing
            with open(os.path.join(labels_dir, f"{doc_name}.txt"), "w") as f:
                for node in doc:
                    if node.class_name in class_list:
                        # Calculate normalized x_center, y_center, width, height
                        x_center = ((node.right - node.left) / 2 + node.left) / img_width
                        y_center = ((node.bottom - node.top) / 2 + node.top) / img_height
                        width = (node.right - node.left) / img_width
                        height = (node.bottom - node.top) / img_height
                        
                        # Write to label file
                        f.write(f"{class_dict[node.class_name]} {x_center} {y_center} {width} {height}\n")
        else:
            anchors = [(randint(0, img_width-1216), randint(0, img_height-1216)) for _ in range(crop_times)]
                
            for idx, anchor in enumerate(anchors):
                x, y = anchor
                dst_path = os.path.join(images_dir, f"{doc_name}_{idx}.png")
                with Image.open(src_path) as img:
                    img = img.crop((x, y, x+1216, y+1216))
                    img.save(dst_path)
                with open(os.path.join(labels_dir, f"{doc_name}_{idx}.txt"), "w") as f:
                    for node in doc:
                        if node.class_name not in class_list or node.bottom > y+1216 or node.top < y or node.left < x or node.right > x+1216:
                            continue
                        # Calculate normalized x_center, y_center, width, height
                        x_center = (((node.right - node.left) / 2 + node.left)-x) / 1216
                        y_center = (((node.bottom - node.top) / 2 + node.top)-y) / 1216
                        width = (node.right - node.left) / 1216
                        height = (node.bottom - node.top) / 1216
                        
                        # Write to label file
                        f.write(f"{class_dict[node.class_name]} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', 
                        default='data/MUSCIMA++/v2.0/data/annotations', help="data directory of annotations")
    parser.add_argument('-i', '--image_dir', default="data/MUSCIMA++/datasets_r_staff/images", help='data directory of staff removed images')
    parser.add_argument('-c', '--classes', default='essential',
                        help="The classes used for training. Possible values are ['essn', 'essential', '20', 'all']. Default to 'essential'.")
    parser.add_argument('--seed', default=314, help="The random seed.")
    parser.add_argument('--save_dir', default='data/MUSCIMA++/datasets_r_staff_essn_crop', help='The output directory.')
    parser.add_argument('--save_config', default='data_staff_removed_crop.yaml', help='The path to save yaml file')
    parser.add_argument('--split_file', default='splits/mob_split.yaml', help='The split yaml file.')
    parser.add_argument('--crop_times', default=14, type=int, help='number of crops for each image. Default to 14.')
                        
    args = parser.parse_args()

    set_seed(args.seed)

    print("Reading annotations...")
    split_file = load_split(args.split_file)
    docs = {}
    for mode in ("train", "valid", "test"):
        cropobject_fnames = [os.path.join(args.data, f) for f in os.listdir(args.data) if os.path.splitext(os.path.basename(f))[0] in split_file[mode]]
        docs[mode] = [read_nodes_from_file(f) for f in cropobject_fnames]
    print("Annotations Loaded.")

    class_list, class_dict = get_classlist_and_classdict(args.classes)

    for mode in "train", "valid", "test":
        print(f"Processing {mode}...")
        generate(docs[mode], class_list, class_dict, args.save_dir, mode, args.crop_times, args.image_dir)
        print("DONE.")

    print("Writing yaml...")
    id2clsname = {idx:clsname for clsname, idx in class_dict.items()}
    max_id = max(id2clsname.keys())
    id2clsname = {i: (id2clsname[i] if i in id2clsname else "NA") for i in range(0, max_id+1)}
    config_to_save = {
        "path": f"../{args.save_dir}",
        "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": id2clsname,
    }
    with open(os.path.join(args.save_dir, args.save_config), 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    
    print("DONE.")
