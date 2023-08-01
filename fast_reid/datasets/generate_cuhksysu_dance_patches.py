import os
import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

def make_parser():
    parser = argparse.ArgumentParser("dancetrack reid dataset")

    parser.add_argument("--data_path", default="datasets", help="path to dancetrack data")
    parser.add_argument("--save_path", default="fast_reid/datasets", help="Path to save the dancetrack-reid dataset")

    return parser

# ============================ for dancetrack ============================
def generate_trajectories(file_path, GroundTrues):
    f = open(file_path, 'r')

    lines = f.read().split('\n')        # list of [n_lines] or [n_objs]
    values = []
    for l in lines:
        split = l.split(',')    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <active>, <category>, <visible_ratio>
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]     # int to float
        values.append(numbers)

    values = np.array(values, np.float_)

    if GroundTrues:     # filter objects
        # values = values[values[:, 6] == 1, :]  # Remove ignore objects, only active objects
        # values = values[values[:, 7] == 1, :]  # Pedestrian only
        values = values[values[:, 8] > 0.4, :]  # visibility only

    values = np.array(values)
    values[:, 4] += values[:, 2]        # tlwh to tlbr
    values[:, 5] += values[:, 3]

    return values

def main_dancetrack(args):
    # NOTE: id starts from 0.
    # Create folder for outputs
    save_path = os.path.join(args.save_path, 'cuhksysu-dancetrack-reid')
    os.makedirs(save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)

    # Get gt data
    data_path = os.path.join(args.data_path, 'dancetrack', 'train')

    seqs = os.listdir(data_path)

    seqs.sort()

    id_offset = 0

    for seq in seqs:        # iteration over seqs
        print("current seq", seq)
        print("current id_offset", id_offset)

        ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
        gt = generate_trajectories(ground_truth_path, GroundTrues=False)  # frame, id, x_tl, y_tl, x_br, y_br, active, category, visible_ratio

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in tqdm(range(num_frames)):     # iteration over frames
            img = cv2.imread(os.path.join(images_path, img_files[f]))
            if img is None:
                print("ERROR: Receive empty frame")
                continue
            H, W, _ = np.shape(img)
            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)     # dets in current frame. [id, x_tl, y_tl, x_br, y_br, active, category, visible_ratio]
            for d in range(np.size(det, 0)):
                id_ = det[d, 0] + 1     # 0-index to 1-index
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]
                # clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]        # crop image

                max_id_per_seq = max(max_id_per_seq, id_)       # update 'max_id_per_seq'

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (str(id_+id_offset)).zfill(7) + '_' + seq[-4:] + '_' + (str(f+1)).zfill(7) + '_acc_data.bmp'


                cv2.imwrite(os.path.join(train_save_path, fileName), patch)

        id_offset += max_id_per_seq
    return id_offset        # just add as above

# ============================ for cuhksysu ============================
def tlwh2xyxy(det, H, W):
    x1 = det[0]
    y1 = det[1]
    x2 = det[0] + det[2]        # tlwh2xyxy
    y2 = det[1] + det[3]
    # clamp
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, W))
    y2 = int(min(y2, H))
    return [x1, y1, x2, y2]

def save_patch(img, det, id, save_path, seq=1, frame=1,):
    x1, y1, x2, y2 = det

    # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
    patch = img[y1:y2, x1:x2, :]  # crop image

    # plt.figure()
    # plt.imshow(patch)
    # plt.show()

    # -000001_5_0000002_acc_data.bmp
    fileName = (str(id)).zfill(7) + '_' + str(seq) + '_' + (str(frame + 1)).zfill(7) + '_acc_data.bmp'

    try:
        cv2.imwrite(os.path.join(save_path, fileName), patch)
    except:
        print('skip box which is too small...')

def main_cuhksysu(args, id_offset, seq_offset=1000):

    # Create folder for outputs
    save_path = os.path.join(args.save_path, 'cuhksysu-dancetrack-reid')
    os.makedirs(save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)

    # Get gt data
    data_path = os.path.join(args.data_path, 'CUHKSYSU')
    anno_path = os.path.join(data_path, 'annotations/train.json')
    img_dir = os.path.join(data_path, 'images')

    with open(anno_path) as f:
        annos = json.load(f)

    for anno in tqdm(annos['annotations']):
        img_file_name = annos['images'][anno['image_id']-1]['file_name'].split('/')[-1]
        W, H = annos['images'][anno['image_id']-1]['width'], annos['images'][anno['image_id']-1]['height']
        seq = int(os.path.basename(img_file_name)[1:-4]) + seq_offset        # + seq_offset in case
        img = cv2.imread(os.path.join(img_dir, img_file_name))
        id = int(anno['track_id']) + id_offset + 1      # + 1 in case 0-index
        det = tlwh2xyxy(anno['bbox'], H, W)

        save_patch(img, det, id, train_save_path, seq=str(seq))
        # cv2.imwrite('img.png', img)

if __name__ == "__main__":
    args = make_parser().parse_args()
    id_offset = main_dancetrack(args)                   # dancetrack
    main_cuhksysu(args, id_offset, seq_offset=1000)     # cuhksysu
