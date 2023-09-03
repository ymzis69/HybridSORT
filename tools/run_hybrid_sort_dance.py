from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluatorDance as MOTEvaluator

from utils.args import make_parser, args_merge_params_form_exp
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True
    rank = args.local_rank
    file_name = os.path.join(exp.output_dir, args.expn)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    result_dir = "{}_test".format(args.expn) + \
                 "_EGWeightHigh" + str(args.EG_weight_high_score) + \
                 "_EGWeightLow" + str(args.EG_weight_low_score) + \
                 "_WithLongTermReIDCorrection" + str(args.with_longterm_reid_correction) + \
                 "_LongTermReIDCorrectionThresh" + str(args.longterm_reid_correction_thresh) + \
                 "_LongTermReIDCorrectionThreshLow" + str(args.longterm_reid_correction_thresh_low) + \
                 "_IoUThresh" + str(args.iou_thresh) + \
                 "_ScoreDifInterval" + str(args.TCM_first_step_weight) + \
                 "_SecScoreDifInterval" + str(args.TCM_byte_step_weight) \
        if args.test else \
        "{}_val".format(args.expn) + \
        "_EGWeightHigh" + str(args.EG_weight_high_score) + \
        "_EGWeightLow" + str(args.EG_weight_low_score) + \
        "_WithLongTermReIDCorrection" + str(args.with_longterm_reid_correction) + \
        "_LongTermReIDCorrectionThresh" + str(args.longterm_reid_correction_thresh) + \
        "_LongTermReIDCorrectionThreshLow" + str(args.longterm_reid_correction_thresh_low) + \
        "_IoUThresh" + str(args.iou_thresh) + \
        "_ScoreDifInterval" + str(args.TCM_first_step_weight) + \
        "_SecScoreDifInterval" + str(args.TCM_byte_step_weight)
    results_folder = os.path.join(file_name, result_dir)
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test, run_tracking=True)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start tracking
    if not args.hybrid_sort_with_reid:
        *_, summary = evaluator.evaluate_hybrid_sort(
            args, model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )
    else:
        *_, summary = evaluator.evaluate_hybrid_sort_reid(
                args, model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )

    
    if args.test:
        # we skip evaluation for inference on test set
        return 

    logger.info("\n" + summary)

    if args.dataset == "dancetrack":
        hota_command = "python3 TrackEval/scripts/run_mot_challenge.py " \
                       "--SPLIT_TO_EVAL val  " \
                       "--METRICS HOTA CLEAR Identity " \
                       "--GT_FOLDER datasets/dancetrack/val " \
                       "--SEQMAP_FILE datasets/dancetrack/val/val_seqmap.txt " \
                       "--SKIP_SPLIT_FOL True " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL True " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--PLOT_CURVES False " \
                       "--TRACKERS_FOLDER " + results_folder
    elif args.dataset == "mot17":
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT17 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/mot/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    elif args.dataset == "mot20":
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT20 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT20/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    else:
        assert args.dataset in ["dancetrack", "mot17"]
    os.system(hota_command)

    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    args_merge_params_form_exp(args, exp)

    if not args.expn:
        args.expn = exp.exp_name
    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )