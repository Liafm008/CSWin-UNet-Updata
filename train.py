import argparse
import os

import torch

from config import get_config
from networks.vision_transformer import CSwinUnet as ViT_seg
from runtime_utils import configure_runtime, resolve_data_dir, resolve_device
from trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/Synapse/train_npz",
    help="root dir for data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
)
parser.add_argument(
    "--num_classes", type=int, default=9, help="output channel of network"
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
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
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
    default=8,
    help="number of dataloader workers",
)
parser.add_argument(
    "--disable_tf32",
    action="store_true",
    help="disable TF32 matmul/cudnn acceleration on Ampere+ GPUs",
)
parser.add_argument(
    "--dice_weight",
    type=float,
    default=0.4,
    help="weight for Dice loss; paper uses 0.4",
)
parser.add_argument(
    "--ce_weight",
    type=float,
    default=0.6,
    help="weight for cross-entropy loss; paper uses 0.6",
)
parser.add_argument(
    "--skip_fusion",
    type=str,
    default="none",
    choices=[
        "none",
        "attention",
        "decoder_gate",
        "sab_cab",
        "dca",
        "sdi",
        "sdi_mid",
        "sdi_resprod",
        "sdi_gate",
        "sdi_add",
    ],
    help="skip connection refinement: none, attention, decoder-guided gate, SAB/CAB, DCA, SDI, mid-level SDI, residual-product SDI, gated SDI, or additive SDI",
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
    "--max_train_batches",
    type=int,
    default=None,
    help="debug only: stop training after this many batches",
)


def main():
    args = parser.parse_args()
    if args.dataset == "Synapse":
        args.root_path = resolve_data_dir(args.root_path, "train_npz")

    config = get_config(args)
    configure_runtime(
        seed=args.seed,
        deterministic=bool(args.deterministic),
        use_tf32=not args.disable_tf32,
    )

    args.device = resolve_device(args.device)
    args.use_amp = args.amp_opt_level != "O0"

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 9,
        },
    }

    if dataset_name not in dataset_config:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    if args.device.type == "cuda":
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 1

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    os.makedirs(args.output_dir, exist_ok=True)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(
        args.device
    )
    if not args.resume:
        net.load_from(config)

    trainer = {"Synapse": trainer_synapse}
    trainer[dataset_name](args, net, args.output_dir)


if __name__ == "__main__":
    main()
