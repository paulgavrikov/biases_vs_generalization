import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from utils import get_normalized_model, accuracy, AverageMeter, assert_imagenet_consistency, str2bool, seed_everything
from tqdm import tqdm
from torchvision.datasets import ImageNet
import pyrtools as pt
from functools import partial
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


NOISE_SDS = [0, 0.02, 0.04, 0.08, 0.16]
SNRS = [0.625, 1.25, 2.5, 5, 10]
N_FREQS = 7
N_NOISES = len(NOISE_SDS)
CONTRAST = 0.2
IMAGENET_MEAN = 0.449
EPSILON = 0.1 # minimum gap between pixel limit and pixel value
IMAGE_SIZE = 224


def rmspower(im):
    """Computes RMS power of an image."""
    return np.sqrt(np.mean(np.square(im)))

def contrast_normalize_np(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def solomon_filter(image, noise_sd, freq, contrast=CONTRAST, imagenet_mean=IMAGENET_MEAN, epsilon=EPSILON, n_freqs=N_FREQS):
    """Applies noise sampled from a Gaussian with mean 0 and given 
    standard deviation to an image at a given spatial frequency."""

    def add_noise_at_sf(image, noise, freq):

        if noise_sd != 0:
            # create a laplacian pyramid of noise with floor(log_2(224)) levels.
            # Levels are octave spaced because their resolutions differ by 
            # a factor of 2
            pyr = pt.pyramids.LaplacianPyramid(noise)

            # reconstruct noise in pyramid for required sf band
            bandi = n_freqs - (freq+1)
            recon = pyr.recon_pyr(levels=bandi)
            first_recon = pyr.recon_pyr(levels=0)
            sf_noise = recon * rmspower(first_recon) / rmspower(recon) # equate power with first noise	
            noisyim = image + sf_noise

            return noisyim, sf_noise

        else:
            # if noise SD is 0, return image
            noisyim = image
            return noisyim, np.zeros_like(image)

    # normalize image to 0-1, decrease histogram width (contrast) and shift to imagenet mean
    image = (image / 255.0)
    image = (image - image.mean()) * contrast + imagenet_mean

    # generate Gaussian noise with mean 0 and given SD
    if noise_sd != 0:
        noise = np.random.randn(*image.shape) * noise_sd
    else:
        noise = None

    # add noise at required sf to image
    noisyim, sf_noise = add_noise_at_sf(image, noise, freq)

    # check for out of bounds values
    if noisyim.min() < 0 or noisyim.max() > 1:
        # find out of bound pixels and distort image appropriately
        for i in range(noisyim.shape[0]):
            for j in range(noisyim.shape[1]):
                OOB = noisyim[i, j] < 0 or noisyim[i, j] > 1

                if OOB:
                    if noisyim[i, j] < 0:
                        newp = epsilon - sf_noise[i, j]
                        otherOOB = newp > 1
                    elif noisyim[i, j] > 1:
                        newp = 1 - epsilon - sf_noise[i, j]
                        otherOOB = newp < 0

                    if otherOOB:
                        # means that shifting image pixel causes out of bounds in other direction
                        # then only soln is to resample noise until no OOB
                        # this will might cause noisy image to be outside desired SF band
                        # but assuming only few noise pixels have to be replaced this way
                        # that should be ok
                        # TODO: check how many pixels we're doing this for
                        newn = sf_noise[i, j]
                        while newn + image[i,j] < 0 or newn + image[i,j] > 1:
                            newn = np.random.randn() * noise_sd
                        sf_noise[i, j] = newn

                    else:
                        # we can just set image pixel to shift value
                        image[i, j] = newp

        # compute new noisy image
        noisyim = image + sf_noise 

        assert noisyim.max() <= 1 and noisyim.min() >= 0, "Image out of bounds"

    return noisyim.astype(np.float32)


def criticalband_transform(x, noise_sd, f_i):
    x = np.asarray(x.convert("L"), dtype=np.uint8)
    x = solomon_filter(x, noise_sd, f_i)
    x = np.stack([x, x, x], axis=2)
    return x


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
        run = wandb.init(project="imagenet_criticalband", name=args.model, config=args)

    device = args.device
    
    model = get_normalized_model(args.model)
    model.to(device)
    
    assert_imagenet_consistency(os.path.join(args.imagenet, "val"))


    for f_i in list(range(N_FREQS)):
        for noise_sd in NOISE_SDS:
            if noise_sd == 0 and f_i != 0:
                continue

            dataset = ImageNet(args.imagenet, 
                                split="val", 
                                transform=torchvision.transforms.Compose(
                                        [
                                            torchvision.transforms.Resize(256), 
                                            torchvision.transforms.CenterCrop(224), 
                                            partial(criticalband_transform, noise_sd=noise_sd, f_i=f_i),
                                            torchvision.transforms.ToTensor(),
                                        ]
                                    )
                                )

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.num_workers)
            top1_acc = eval_loader_top1(model, dataloader, device)

            if run:
                run.log({f"top1_acc/criticalband/f={f_i}/sd={noise_sd}": top1_acc})

            print(f"Top-1 accuracy for {args.model} with noise SD {noise_sd} and frequency {f_i} is {top1_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/workspace/data/datasets/imagenet/")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--wandb', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    
    seed_everything(args.seed)

    main(args)
