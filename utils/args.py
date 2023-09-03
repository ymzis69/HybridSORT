import argparse

def make_parser():
    parser = argparse.ArgumentParser("OC-SORT parameters")
    parser.add_argument("--expn", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(
        "-f", "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16", dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")

    parser.add_argument("--TCM_first_step", default=False, action="store_true", help="use TCM in first step.")
    parser.add_argument("--TCM_byte_step", default=False, action="store_true", help="use TCM in byte step.")
    parser.add_argument("--TCM_first_step_weight", type=float, default=1.0, help="TCM first step weight")
    parser.add_argument("--TCM_byte_step_weight", type=float, default=1.0, help="TCM second step weight")
    parser.add_argument("--hybrid_sort_with_reid", default=False, action="store_true", help="use ReID model for Hybrid SORT.")

    # for fast reid
    parser.add_argument("--EG_weight_high_score", default=0.0, type=float, help="weight of appearance cost matrix when using EG")
    parser.add_argument("--EG_weight_low_score", default=0.0, type=float, help="weight of appearance cost matrix when using EG")
    parser.add_argument("--low_thresh", default=0.1, type=float, help="threshold of low score detections for BYTE")
    parser.add_argument("--high_score_matching_thresh", default=0.8, type=float, help="matching threshold for detections with high score")
    parser.add_argument("--low_score_matching_thresh", default=0.5, type=float, help="matching threshold for detections with low score")
    parser.add_argument("--alpha", default=0.8, type=float, help="momentum of embedding update")
    parser.add_argument("--with_fastreid", dest="with_fastreid", default=False, action="store_true", help="use FastReID flag.")
    parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"fast_reid/logs/CUHKSYSU_DanceTrack/sbs_S50/model_final.pth", type=str, help="reid weight path")
    parser.add_argument("--with_longterm_reid", dest="with_longterm_reid", default=False, action="store_true", help="use long-term reid features for association.")
    parser.add_argument("--longterm_reid_weight", default=0.0, type=float, help="weight of appearance cost matrix when using long term reid features in 1st stage association")
    parser.add_argument("--longterm_reid_weight_low", default=0.0, type=float, help="weight of appearance cost matrix when using long term reid features in 2nd stage association")
    parser.add_argument("--with_longterm_reid_correction", dest="with_longterm_reid_correction", default=False, action="store_true", help="use long-term reid features for association correction.")
    parser.add_argument("--longterm_reid_correction_thresh", default=1.0, type=float, help="threshold of correction when using long term reid features in 1st stage association")
    parser.add_argument("--longterm_reid_correction_thresh_low", default=1.0, type=float, help="threshold of correction when using long term reid features in 2nd stage association")
    parser.add_argument("--longterm_bank_length", type=int, default=30, help="max length of reid feat bank")
    parser.add_argument("--adapfs", dest="adapfs", default=False, action="store_true", help="Adaptive Feature Smoothing.")
    # ECC for CMC
    parser.add_argument("--ECC", dest="ECC", default=False, action="store_true", help="use ECC for CMC.")

    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--out_path', type=str, help="path to save output results")
    parser.add_argument("--dataset", type=str, default="mot17", help="kitti or bdd")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    # for demo video
    parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument( "--path", default="./videos/demo.mp4", help="path to images or video")
    parser.add_argument("--demo_dancetrack", default=False, action="store_true",
                        help="only for dancetrack demo, replace timestamp with dancetrack sequence name.")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser


def args_merge_params_form_exp(args,exp):
    for k, v in exp.__dict__.items():
        if k in args.__dict__:
            (args.__dict__)[k] = v