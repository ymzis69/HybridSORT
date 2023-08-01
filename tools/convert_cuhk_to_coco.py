"""convert cuhk from label_with_ids in JDE to coco"""
import os
import numpy as np
import json
from PIL import Image

input_root_dataset_folder = "datasets/"
output_root_dataset_folder = "datasets/"
original_dataset_folder = "datasets/CUHKSYSU"
output_dataset_folder = "datasets/CUHKSYSU"

original_image_path = os.path.join(original_dataset_folder, "images")
image_names = os.listdir(original_image_path)

with open(os.path.join(output_root_dataset_folder, 'cuhksysu.train'), 'w') as f:
    for name in image_names:
        full = 'CUHKSYSU/images/'+name+'\n'
        f.write(full)

data_file_path = os.path.join(output_root_dataset_folder, 'cuhksysu.train')
out_path = os.path.join(output_dataset_folder, 'annotations')


def load_paths(data_path):
    with open(data_path, 'r') as file:
        img_files = file.readlines()
        img_files = [x.replace('\n', '') for x in img_files]
        img_files = list(filter(lambda x: len(x) > 0, img_files))
    label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in img_files]
    # print(img_files)
    return img_files, label_files

if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, 'train.json')
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
    img_paths, label_paths = load_paths(data_file_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for img_path, label_path in zip(img_paths, label_paths):
        image_cnt += 1
        # print(os.path.join("../datasets", img_path))
        im = Image.open(os.path.join(input_root_dataset_folder, img_path))

        image_info = {'file_name': img_path,
                      'id': image_cnt,
                      'height': im.size[1],
                      'width': im.size[0],
                      'video_id': 1,
                      'frame_id': 1,
                      'prev_image_id': image_cnt - 1,
                      'prev_image_id': image_cnt + 1
                      }
        out['images'].append(image_info)
        # Load labels
        if os.path.isfile(os.path.join(input_root_dataset_folder, label_path)):
            labels0 = np.loadtxt(os.path.join(input_root_dataset_folder, label_path), dtype=np.float32).reshape(-1, 6)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = image_info['width'] * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = image_info['height'] * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = image_info['width'] * labels0[:, 4]
            labels[:, 5] = image_info['height'] * labels0[:, 5]
        else:
            labels = np.array([])
        for i in range(len(labels)):
            ann_cnt += 1
            labels[i, 1] = int(labels[i, 1])
            if float(labels[i, 1]) != -1:       # min track_id from 0 to 1
                labels[i, 1] = int(labels[i, 1]+1)
            fbox = labels[i, 2:6].tolist()
            ann = {'id': ann_cnt,
                   'category_id': 1,
                   'image_id': image_cnt,
                   'track_id': int(labels[i, 1]),
                   'bbox': fbox,
                   'area': fbox[2] * fbox[3],
                   'iscrowd': 0}
            out['annotations'].append(ann)
    print('loaded train for {} images and {} samples'.format(len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


# print('crowdhuman_val')
