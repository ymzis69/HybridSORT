# Hybrid-SORT

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

#### Hybrid-SORT is a simply and strong multi-object tracker.

> [**Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking**](https://arxiv.org/abs/2308.00783)
> 

## Abstract

Multi-Object Tracking (MOT) aims to detect and associate all desired objects across frames. Most methods accomplish the task by explicitly or implicitly leveraging strong cues (i.e., spatial and appearance information), which exhibit powerful instance-level discrimination. However, when object occlusion and clustering occur, both spatial and appearance information will become ambiguous simultaneously due to the high overlap between objects. In this paper, we demonstrate that this long-standing challenge in MOT can be efficiently and effectively resolved by incorporating weak cues to compensate for strong cues. Along with velocity direction, we introduce the confidence state and height state as potential weak cues. With superior performance, our method still maintains Simple, Online and Real-Time (SORT) characteristics. Furthermore, our method shows strong generalization for diverse trackers and scenarios in a plug-and-play and training-free manner. Significant and consistent improvements are observed when applying our method to 5 different representative trackers. Further, by leveraging both strong and weak cues, our method Hybrid-SORT achieves superior performance on diverse benchmarks, including MOT17, MOT20, and especially DanceTrack where interaction and occlusion are frequent and severe.

### Highlights

- Hybrid-SORT is a **SOTA** heuristic trackers on DanceTrack and performs excellently on MOT17/MOT20 datasets.
- Maintains **Simple, Online and Real-Time (SORT)** characteristics.
- **Training-free** and **plug-and-play** manner.
- **Strong generalization** for diverse trackers and scenarios

### Pipeline

<center>
<img src="assets/pipeline.png" width="800"/>
</center>

## News
* [12/09/2023]: Hybrid-SORT is accepted by **AAAI2024**!
* [08/24/2023]: Hybrid-SORT is supported in [yolo_tracking](https://github.com/mikel-brostrom/yolo_tracking). Many thanks to [@mikel-brostrom](https://github.com/mikel-brostrom) for the contribution.
* [08/01/2023]: The [arxiv preprint](https://arxiv.org/abs/2308.00783) of Hybrid-SORT is released.

## Tracking performance

### Results on DanceTrack test set

| Tracker          | HOTA | MOTA | IDF1 | FPS  |
| :--------------- | :--: | :--: | :--: | :--: |
| OC-SORT          | 54.6 | 89.6 | 54.6 | 30.3 |
| Hybrid-SORT      | 62.2 | 91.6 | 63.0 | 27.8 |
| Hybrid-SORT-ReID | 65.7 | 91.8 | 67.4 | 15.5 |

### Results on MOT20 challenge test set

| Tracker          | HOTA | MOTA | IDF1 |
| :--------------- | :--: | :--: | :--: |
| OC-SORT          | 62.1 | 75.5 | 75.9 |
| Hybrid-SORT      | 62.5 | 76.4 | 76.2 |
| Hybrid-SORT-ReID | 63.9 | 76.7 | 78.4 |

### Results on MOT17 challenge test set

| Tracker          | HOTA | MOTA | IDF1 |
| :--------------- | :--: | :--: | :--: |
| OC-SORT          | 63.2 | 78.0 | 77.5 |
| Hybrid-SORT      | 63.6 | 79.3 | 78.4 |
| Hybrid-SORT-ReID | 64.0 | 79.9 | 78.7 |

## Installation

Hybrid-SORT code is based on [OC-SORT](https://github.com/noahcao/OC_SORT) and [FastReID](https://github.com/JDAI-CV/fast-reid). The ReID component is optional and based on [FastReID](https://github.com/JDAI-CV/fast-reid). Tested the code with Python 3.8 + Pytorch 1.10.0 + torchvision 0.11.0.

Step1. Install Hybrid_SORT

```shell
git clone https://github.com/ymzis69/HybridSORT.git
cd HybridSORT
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```shell
pip3 install cython_bbox pandas xmltodict
```

Step4. [optional] FastReID Installation

You can refer to [FastReID Installation](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

```shell
pip install -r fast_reid/docs/requirements.txt
```

## Data preparation

**Our data structure is the same as [OC-SORT](https://github.com/noahcao/OC_SORT).** 

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [DanceTrack](https://github.com/DanceTrack/DanceTrack), [CUHKSYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) and put them under <HYBRIDSORT_HOME>/datasets in the following structure (CrowdHuman, Cityperson and ETHZ are not needed if you download YOLOX weights from [ByteTrack](https://github.com/ifzhang/ByteTrack) or [OC-SORT](https://github.com/noahcao/OC_SORT)) :

   ```
   datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |        └——————Crowdhuman_train
   |        └——————Crowdhuman_val
   |        └——————annotation_train.odgt
   |        └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Cityscapes
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
   |        └——————eth01
   |        └——————...
   |        └——————eth07
   └——————CUHKSYSU
   |        └——————images
   |        └——————labels_with_ids
   └——————dancetrack        
            └——————train
               └——————train_seqmap.txt
            └——————val
               └——————val_seqmap.txt
            └——————test
               └——————test_seqmap.txt
   ```

2. Prepare DanceTrack dataset:

   ```python
   # replace "dance" with ethz/mot17/mot20/crowdhuman/cityperson/cuhk for others
   python3 tools/convert_dance_to_coco.py 
   ```

3. Prepare MOT17/MOT20 dataset. 

   ```python
   # build mixed training sets for MOT17 and MOT20 
   python3 tools/mix_data_{ablation/mot17/mot20}.py
   ```

4. [optional] Prepare ReID datasets:

   ```
   cd <HYBRIDSORT_HOME>
   
   # For MOT17 
   python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 17
   
   # For MOT20
   python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 20
   
   # For DanceTrack
   python3 fast_reid/datasets/generate_cuhksysu_dance_patches.py --data_path <dataets_dir> 
   ```

## Model Zoo

Download and store the trained models in 'pretrained' folder as follow:

```
<HYBRIDSORT_HOME>/pretrained
```

### Detection Model

We provide some pretrained YOLO-X weights for Hybrid-SORT, which are inherited from [ByteTrack](https://github.com/ifzhang/ByteTrack).

| Dataset         | HOTA | IDF1 | MOTA | Model                                                        |
| --------------- | ---- | ---- | ---- | ------------------------------------------------------------ |
| DanceTrack-val  | 59.3 | 60.6 | 89.5 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| DanceTrack-test | 62.2 | 63.0 | 91.6 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-half-val  | 67.1 | 78.0 | 75.8 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-test      | 63.6 | 78.7 | 79.9 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT20-test      | 62.5 | 78.4 | 76.7 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |


* For more YOLO-X weights, please refer to the model zoo of [ByteTrack](https://github.com/ifzhang/ByteTrack).

### ReID Model

Ours ReID models for **MOT17/MOT20** is the same as [BoT-SORT](https://github.com/NirAharon/BOT-SORT) , you can download from [MOT17-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), ReID models for DanceTrack is trained by ourself, you can download from [DanceTrack](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing).

**Notes**:


* [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) is trained by [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), because the weight from BOT-SORT is corrupted. Refer to [Issue](https://github.com/GerardMaggiolino/Deep-OC-SORT/issues/6).
* ReID models for DanceTrack is trained by ourself, with both DanceTrack and CUHKSYSU datasets.

## Training

### Train the Detection Model

You can use Hybrid-SORT without training by adopting existing detectors. But we borrow the training guidelines from ByteTrack in case you want work on your own detector. 

Download the COCO-pretrained YOLOX weight [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) and put it under *\<HYBRIDSORT_HOME\>/pretrained*.

* **Train ablation model (MOT17 half train and CrowdHuman)**

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train MOT17 test model (MOT17 train, CrowdHuman, Cityperson and ETHZ)**

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train MOT20 test model (MOT20 train, CrowdHuman)**

  For MOT20, you need to uncomment some code lines to add box clipping: [[1]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment),[[2]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122),[[3]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217) and [[4]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115). Then run the command:

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train on DanceTrack train set**

  ```shell
  python3 tools/train.py -f exps/example/dancetrack/yolox_x.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train custom dataset**

  First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

  ```shell
  python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

### Train the ReID Model

After generating MOT ReID dataset as described in the 'Data Preparation' section.

```shell
cd <BoT-SORT_dir>

# For training MOT17 
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT17/sbs_S50.yml MODEL.DEVICE "cuda:0"

# For training MOT20
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT20/sbs_S50.yml MODEL.DEVICE "cuda:0"

# For training DanceTrack, we joint the CHUKSUSY to train ReID Model for DanceTrack
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml MODEL.DEVICE "cuda:0"
```

Refer to [FastReID](https://github.com/JDAI-CV/fast-reid)  repository for addition explanations and options.

## Tracking

**Notes**:


* Some parameters are set in the cfg.py. For example, if you run Hybrid-SORT on the dancetrack-val dataset, you should pay attention to the line 35-45 in ```exps/example/mot/yolox_dancetrack_val_hybrid_sort.py``` .
* We set  ```fp16==False``` on the MOT datasets becacuse fp16 will lead to significant result fluctuations.

### DanceTrack

**dancetrack-val dataset**

```
# Hybrid-SORT
python tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py -b 1 -d 1 --fp16 --fuse --expn $exp_name 

# Hybrid-SORT-ReID
python tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_dancetrack_val_hybrid_sort_reid.py -b 1 -d 1 --fp16 --fuse --expn $exp_name
```

**dancetrack-test dataset**

```
# Hybrid-SORT
python tools/run_hybrid_sort_dance.py --test -f exps/example/mot/yolox_dancetrack_test_hybrid_sort.py -b 1 -d 1 --fp16 --fuse --expn $exp_name

# Hybrid-SORT-ReID
python tools/run_hybrid_sort_dance.py --test -f exps/example/mot/yolox_dancetrack_test_hybrid_sort_reid.py -b 1 -d 1 --fp16 --fuse --expn $exp_name
```

### MOT20

**MOT20-test dataset**

```
#Hybrid-SORT
python tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_mix_mot20_ch_hybrid_sort.py -b 1 -d 1 --fuse --mot20 --expn $exp_name 

#Hybrid-SORT-ReID
python tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_mix_mot20_ch_hybrid_sort_reid.py -b 1 -d 1 --fuse --mot20 --expn $exp_name
```

Hybrid-SORT is designed for online tracking, but offline interpolation has been demonstrated efficient for many cases and used by other online trackers. If you want to reproduct out result on  **MOT20-test** dataset, please use the linear interpolation over existing tracking results:

```shell
# offline post-processing
python3 tools/interpolation.py $result_path $save_path
```

### MOT17

**MOT17-val dataset**

```
# Hybrid-SORT
python3 tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_ablation_hybrid_sort.py -b 1 -d 1 --fuse --expn $exp_name 

# Hybrid-SORT-ReID
python3 tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_ablation_hybrid_sort_reid.py -b 1 -d 1 --fuse --expn  $exp_name 
```

**MOT17-test dataset**

```
# Hybrid-SORT
python3 tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_mix_det_hybrid_sort.py -b 1 -d 1 --fuse --expn $exp_name

# Hybrid-SORT-ReID
python3 tools/run_hybrid_sort_dance.py -f exps/example/mot/yolox_x_mix_det_hybrid_sort_reid.py -b 1 -d 1 --fuse --expn $exp_name
```

Hybrid-SORT is designed for online tracking, but offline interpolation has been demonstrated efficient for many cases and used by other online trackers. If you want to reproduct out result on  **MOT17-test** dataset, please use the linear interpolation over existing tracking results:

```shell
# offline post-processing
python3 tools/interpolation.py $result_path $save_path
```

### Demo

Hybrid-SORT, with the parameter settings of the dancetrack-val dataset

```
python3 tools/demo_track.py --demo_type image -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py -c pretrained/ocsort_dance_model.pth.tar --path ./datasets/dancetrack/val/dancetrack0079/img1 --fp16 --fuse --save_result
```

Hybrid-SORT-ReID, with the parameter settings of the dancetrack-val dataset

```
python3 tools/demo_track.py --demo_type image -f exps/example/mot/yolox_dancetrack_val_hybrid_sort_reid.py -c pretrained/ocsort_dance_model.pth.tar --path ./datasets/dancetrack/val/dancetrack0079/img1 --fp16 --fuse --save_result
```

<img src="assets/demo.gif" alt="demo" style="zoom:34%;" />

## TCM on other trackers

download ReID weight from [googlenet_part8_all_xavier_ckpt_56.h5](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) for MOTDT and DeepSORT.

**dancetrack-val dataset**

```
# SORT
python tools/run_sort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --dataset dancetrack --expn sort_score_kalman_fir_step --TCM_first_step

# MOTDT
python3 tools/run_motdt_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --dataset dancetrack --expn motdt_score_kalman_fir_step --TCM_first_step

# ByteTrack
python3 tools/run_byte_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --dataset dancetrack --expn byte_score_kalman_fir_step --TCM_first_step

# DeepSORT
python3 tools/run_deepsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --dataset dancetrack --expn deepsort_score_kalman_fir_step --TCM_first_step
```

**mot17-val dataset**

```
# SORT
python3 tools/run_sort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/ocsort_mot17_ablation.pth.tar -b 1 -d 1 --fuse --expn mot17_sort_score_test_fp32 --TCM_first_step

# MOTDT
python3 tools/run_motdt.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/ocsort_mot17_ablation.pth.tar -b 1 -d 1 --fuse --expn mot17_motdt_score_test_fp32 --TCM_first_step

# ByteTrack
python3 tools/run_byte.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/ocsort_mot17_ablation.pth.tar -b 1 -d 1 --fuse --expn mot17_byte_score_test_fp32 --TCM_first_step --TCM_first_step_weight 0.6

# DeepSORT
python3 tools/run_deepsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/ocsort_mot17_ablation.pth.tar -b 1 -d 1 --fuse --expn mot17_deepsort_score_test_fp32 --TCM_first_step
```

## Citation

If you find this work useful, please consider to cite our paper:
```
@inproceedings{yang2024hybrid,
  title={Hybrid-sort: Weak cues matter for online multi-object tracking},
  author={Yang, Mingzhan and Han, Guangxin and Yan, Bin and Zhang, Wenhua and Qi, Jinqing and Lu, Huchuan and Wang, Dong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6504--6512},
  year={2024}
}
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BOT-SORT) and [FastReID](https://github.com/JDAI-CV/fast-reid). Many thanks for their wonderful works.

