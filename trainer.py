import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from runtime_utils import (
    create_grad_scaler,
    get_autocast,
    load_checkpoint,
    save_checkpoint,
)
from utils import DiceLoss


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import RandomGenerator, Synapse_dataset

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = args.device
    base_lr = args.base_lr
    num_classes = args.num_classes
    dice_weight = getattr(args, "dice_weight", 0.4)
    ce_weight = getattr(args, "ce_weight", 0.6)
    effective_gpu_count = max(1, args.n_gpu)
    batch_size = args.batch_size * effective_gpu_count

    if args.dataset == "Synapse":
        db_train = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose(
                [RandomGenerator(output_size=[args.img_size, args.img_size])]
            ),
        )
    print("The length of train set is: {}".format(len(db_train)))

    num_workers = min(getattr(args, "num_workers", 8), os.cpu_count() or 1)
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    use_amp = bool(getattr(args, "use_amp", False) and device.type == "cuda")
    scaler = create_grad_scaler(use_amp)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    start_epoch = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(
        "{} iterations per epoch. {} max iterations ".format(
            len(trainloader), max_iterations
        )
    )
    best_performance = 0.0

    if args.resume:
        checkpoint, state_dict = load_checkpoint(args.resume, device)
        msg = model.load_state_dict(state_dict, strict=True)
        logging.info("resume model: %s", msg)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        iter_num = checkpoint.get("iter_num", 0)
        best_performance = checkpoint.get("best_performance", 0.0)
        logging.info(
            "resumed from %s at epoch %d, iter %d",
            args.resume,
            start_epoch,
            iter_num,
        )

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].to(device, non_blocking=True)
            label_batch = sampled_batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with get_autocast(device, enabled=use_amp):
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = dice_weight * loss_dice + ce_weight * loss_ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss_ce", loss_ce.item(), iter_num)

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f"
                % (iter_num, loss.item(), loss_ce.item())
            )

            if iter_num % 20 == 0 and image_batch.size(0) > 0:
                sample_index = min(1, image_batch.size(0) - 1)
                image = image_batch[sample_index, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image("train/Image", image, iter_num)
                prediction = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True
                )
                writer.add_image(
                    "train/Prediction", prediction[sample_index, ...] * 50, iter_num
                )
                labs = label_batch[sample_index, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            save_checkpoint(
                save_mode_path,
                model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch_num,
                iter_num=iter_num,
                best_performance=best_performance,
            )
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            save_checkpoint(
                save_mode_path,
                model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch_num,
                iter_num=iter_num,
                best_performance=best_performance,
            )
            best_model_path = os.path.join(snapshot_path, "best_model.pth")
            save_checkpoint(
                best_model_path,
                model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch_num,
                iter_num=iter_num,
                best_performance=best_performance,
            )
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
