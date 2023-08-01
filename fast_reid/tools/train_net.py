#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

# python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/Market1501/sbs_S50.yml --num-gpus 8
# python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/Market1501/sbs_S50.yml --eval-only \
# MODEL.WEIGHTS logs/market1501/sbs_S50/model_final.pth MODEL.DEVICE "cuda:0"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT17/sbs_S50.yml --num-gpus 1
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT20/sbs_S50.yml --num-gpus 1
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/DanceTrack/sbs_S50.yml --num-gpus 1
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/CUHKSYSU/sbs_S50.yml --num-gpus 1
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml --num-gpus 1

# python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/CUHKSYSU_MOT17/sbs_S50.yml MODEL.DEVICE "cuda:0"
# python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/DanceTrack/sbs_S50.yml MODEL.DEVICE "cuda:0"
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
