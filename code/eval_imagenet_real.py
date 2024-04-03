import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, AverageMeter, str2bool, TEST_TRANSFORMS
import json
import os
import torchvision
from torch.utils.data import DataLoader

try:
    import wandb

    HAS_WANDB = True
except:
    HAS_WANDB = False


class ImageNetWithFileNo(torchvision.datasets.ImageNet):

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        file_no = int(path.replace(".JPEG", "").split("_")[-1])

        return sample, target, file_no


def main(args):
    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="imagenet_ReaL", name=args.model, config=args)

    device = args.device

    dataset = ImageNetWithFileNo(args.imagenet, split="val", transform=TEST_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    real_path = os.path.join(args.imagenet, "real.json")
    if not os.path.exists(real_path):
        raise ValueError(
            "Real labels not found. Please download https://github.com/google-research/reassessed-imagenet/blob/master/real.json and place it in the imagenet folder. You set the root to {args.imagenet}"
        )

    with open(real_path, "r") as f:
        real_labels = json.load(f)

    model = get_normalized_model(args.model)
    model.to(device)

    top1_meter = AverageMeter()

    with torch.no_grad():

        progress = tqdm(dataloader)

        for x, _, file_nos in progress:
            bx = x.to(device)

            file_nos = file_nos - 1  # file_nos are 1-indexed

            logits = model(bx)

            predictions = torch.argmax(logits, -1)
            is_correct = [
                pred in real_labels[file_no]
                for pred, file_no in zip(predictions, file_nos)
                if real_labels[file_no]
            ]
            top1 = 100 * sum(is_correct) / len(is_correct)
            samples = len(is_correct)

            top1_meter.update(top1, samples)

            progress.set_postfix({"top1": top1_meter.avg})

    print(f"Accuracy top1: {top1_meter.avg:.2f}%")

    if run:
        run.log({"top1_acc/imagenet-real": top1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--imagenet", type=str, default="/workspace/data/datasets/imagenet/"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--wandb", type=str2bool, default=True)

    args = parser.parse_args()

    main(args)
