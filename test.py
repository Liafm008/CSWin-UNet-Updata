import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import CSwinUnet as ViT_seg
from runtime_utils import (
    configure_runtime,
    load_checkpoint,
    resolve_data_dir,
    resolve_device,
)
from utils import test_single_volume

try:
    from thop import clever_format, profile
except ImportError:
    clever_format = None
    profile = None


parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="../data/Synapse/test_vol_h5",
    help="root dir for validation volume data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument(
    "--num_classes", type=int, default=9, help="output channel of network"
)
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
)
parser.add_argument("--output_dir", type=str, required=True, help="output dir")
parser.add_argument(
    "--max_iterations", type=int, default=30000, help="maximum epoch number to train"
)
parser.add_argument(
    "--max_epochs", type=int, default=150, help="maximum epoch number to train"
)
parser.add_argument(
    "--batch_size", type=int, default=24, help="batch_size per gpu"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument(
    "--is_savenii", action="store_true", help="whether to save results during inference"
)
parser.add_argument(
    "--test_save_dir", type=str, default="../predictions", help="saving prediction as nii"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
)
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs="+",
)
parser.add_argument(
    "--zip",
    action="store_true",
    help="use zipped dataset instead of folder dataset",
)
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="dataset cache mode",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument(
    "--accumulation-steps", type=int, help="gradient accumulation steps"
)
parser.add_argument(
    "--use-checkpoint",
    action="store_true",
    help="whether to use gradient checkpointing to save memory",
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O0",
    choices=["O0", "O1", "O2"],
    help="kept for backward compatibility; O0 disables amp, O1/O2 enable amp",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument(
    "--throughput", action="store_true", help="Test throughput only"
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="device spec such as cuda, cuda:0 or cpu",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="number of dataloader workers",
)
parser.add_argument(
    "--disable_tf32",
    action="store_true",
    help="disable TF32 matmul/cudnn acceleration on Ampere+ GPUs",
)
parser.add_argument(
    "--skip_profile",
    action="store_true",
    help="skip THOP FLOPs/params profiling after inference",
)
parser.add_argument(
    "--skip_fusion",
    type=str,
    default="none",
    choices=["none", "attention", "sdi", "sdi_add"],
    help="skip connection refinement used by the trained checkpoint",
)
parser.add_argument(
    "--sdi_channels",
    type=int,
    default=32,
    help="intermediate channels used by SDI skip fusion",
)
parser.add_argument(
    "--skip_fusion_scale",
    type=float,
    default=0.1,
    help="initial residual scale for skip fusion refinement",
)
parser.add_argument(
    "--postprocess_lcc",
    action="store_true",
    help="keep the largest connected component for each foreground class",
)
parser.add_argument(
    "--postprocess_min_size",
    type=int,
    default=0,
    help="remove connected components smaller than this voxel count",
)


def inference(args, model, test_save_path=None):
    if args.dataset == "Synapse":
        db_test = args.Dataset(
            base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir
        )
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, os.cpu_count() or 1),
        pin_memory=(args.device.type == "cuda"),
    )
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image, label = sampled_batch["image"], sampled_batch["label"]
        case_name = sampled_batch["case_name"][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
            device=args.device,
            use_amp=args.use_amp,
            postprocess_lcc=args.postprocess_lcc,
            postprocess_min_size=args.postprocess_min_size,
        )
        metric_list += np.array(metric_i)
        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (
                i_batch,
                case_name,
                np.mean(metric_i, axis=0)[0],
                np.mean(metric_i, axis=0)[1],
            )
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            "Mean class %d mean_dice %f mean_hd95 %f"
            % (i, metric_list[i - 1][0], metric_list[i - 1][1])
        )
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
        "Testing performance in best val model: mean_dice : %f mean_hd95 : %f"
        % (performance, mean_hd95)
    )
    return "Testing Finished!"


def maybe_profile_model(args, model):
    if args.skip_profile or profile is None or clever_format is None:
        return
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", flops)
    print("Params:", params)


def main():
    args = parser.parse_args()
    if args.dataset == "Synapse":
        args.volume_path = resolve_data_dir(args.volume_path, "test_vol_h5")

    config = get_config(args)
    configure_runtime(
        seed=args.seed,
        deterministic=bool(args.deterministic),
        use_tf32=not args.disable_tf32,
    )

    args.device = resolve_device(args.device)
    args.use_amp = args.amp_opt_level != "O0"

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "volume_path": args.volume_path,
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 9,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(
        args.device
    )

    snapshot = os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "epoch_" + str(args.max_epochs - 1))
    _, state_dict = load_checkpoint(snapshot, args.device)
    msg = net.load_state_dict(state_dict, strict=True)
    print("self trained cswin unet", msg)
    snapshot_name = os.path.basename(snapshot)

    log_folder = "./test_log"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, snapshot_name + ".txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)
    maybe_profile_model(args, net)


if __name__ == "__main__":
    main()
