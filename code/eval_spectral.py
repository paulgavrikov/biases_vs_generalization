import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from utils import get_normalized_model, accuracy, AverageMeter, assert_imagenet_consistency, str2bool
from functools import partial
from tqdm import tqdm
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


def bandpass_filter(bx, cutoff_freq, lowpass=True):
    assert cutoff_freq >= 0 and cutoff_freq <= 1, "cutoff must be in [0, 1]"
    fft = torch.fft.fftshift(torch.fft.fft2(bx))

    if not lowpass:
        cutoff_freq = 1 - cutoff_freq
    
    h, w = fft.shape[-2:]  # height and width
    cy, cx = h // 2, w // 2  # center y, center x
    ry, rx = int(cutoff_freq * cy), int(cutoff_freq * cx)
    
    if lowpass:
        mask = torch.zeros_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 1
    else:
        mask = torch.ones_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 0


    fft = torch.fft.ifft2(torch.fft.ifftshift(fft * mask)).real.clip(0, 1)
    return fft


def eval_loader_top1(model, dataloader, device):
    top1_meter = AverageMeter()

    with torch.no_grad():

        for x, y in (dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            
            top1 = accuracy(logits, y, topk=(1,))[0]
            top1_meter.update(top1.item(), x.size(0))

    return top1_meter.avg



def main(args):
    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="imagenet_spectral", name=args.model, config=args)

    device = args.device
    
    model = get_normalized_model(args.model)
    model.to(device)

    text = ""

    for lowpass in [False, True]:
        tag = 'lowpass' if lowpass else 'highpass'
        text += f"{tag}"
        for cutoff_freq in tqdm([0.01, *np.arange(0.1, 0.99, 0.1), 0.99], desc=tag):
            assert_imagenet_consistency(os.path.join(args.imagenet, "val"))
            dataset = torchvision.datasets.ImageNet(args.imagenet, 
                                                    split="val", 
                                                    transform=torchvision.transforms.Compose(
                                                            [
                                                                torchvision.transforms.Resize(256), 
                                                                torchvision.transforms.CenterCrop(224), 
                                                                torchvision.transforms.ToTensor(),
                                                                partial(bandpass_filter, cutoff_freq=cutoff_freq, lowpass=lowpass)
                                                            ]
                                                        )
                                                    )

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
            top1 = eval_loader_top1(model, dataloader, device)
            print(f"{tag}/{cutoff_freq} accuracy top1: {top1:.2f}%")
            if run:
                run.log(
                    {
                        f"top1_acc/{tag}/{cutoff_freq}": top1
                    }
                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/workspace/data/datasets/imagenet/")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', type=str2bool, default=True)

    args = parser.parse_args()

    main(args)
