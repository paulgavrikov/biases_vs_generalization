import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, accuracy, AverageMeter, str2bool, get_imagenet_folder_loader, assert_imagenet_consistency
import os
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


def find_run_id_by_name(project, name):
    """
    Returns the first run id of the run with the given name.
    Note that there might be multiple runs with the same name.
    Returns None if no run with the given name is found.
    """

    api = wandb.Api()
    runs = api.runs(project)

    for run in runs:
        if run.state == "finished" and run.name == name:
            return run.id

    return None


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', # noise
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', # blur
    'snow', 'frost', 'fog', 'brightness', # weather
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', # digital
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate' # extra
]

SEVERITIES = [1, 2, 3, 4, 5]

def main(args):

    run = None
    if HAS_WANDB and args.wandb:
        if args.wandb_resume:
            id = find_run_id_by_name("imagenet_c", name=args.model)
            assert id is not None, f"Could not find run with name {args.model}"
            print(f"Resuming run {id}")
            run = wandb.init(project="imagenet_c", resume=True, id=id)
        else:
            run = wandb.init(project="imagenet_c", name=args.model, config=args)

    device = args.device
    
    model = get_normalized_model(args.model)
    model.to(device)

    global_top1_meter = AverageMeter()
    global_top5_meter = AverageMeter()

    selected_corruptions = args.corruptions.split(",") if args.corruptions is not None else CORRUPTIONS
    selected_severities = args.severities.split(",") if args.severities is not None else SEVERITIES

    tests = 0
    total_tests = len(selected_corruptions) * len(selected_severities)
    
    for corruption in selected_corruptions:
        for severity in selected_severities:

            tests += 1

            c_path = os.path.join(args.dataset_path, f"{corruption}/{severity}/")
            assert_imagenet_consistency(c_path)

            dataloader = get_imagenet_folder_loader(path=c_path, batch_size=args.batch_size, num_workers=args.num_workers, resize_crop=False)

            top1_meter = AverageMeter()
            top5_meter = AverageMeter()

            with torch.no_grad():

                for x, y in tqdm(dataloader, desc=f"({tests}/{total_tests}) {corruption}/{severity}"):
                    bx = x.to(device)
                    by = y.to(device)

                    # assert that bx is not normalized by mean and std
                    assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

                    logits = model(bx)
                    
                    top1, top5 = accuracy(logits, by, topk=(1, 5))
                    top1_meter.update(top1.item(), bx.size(0))
                    top5_meter.update(top5.item(), bx.size(0))

            global_top1_meter.update(top1_meter.avg)
            global_top5_meter.update(top5_meter.avg)
            
            print(f"{corruption}/{severity} - top1: {top1_meter.avg:.2f}%, top5: {top5_meter.avg:.2f}%")
            if run:
                run.log(
                    {
                        f"top1_acc/{corruption}_{severity}": top1_meter.avg,
                        f"top5_acc/{corruption}_{severity}": top5_meter.avg,
                    }
                )
    
    print(f"Average Accuracy top1: {global_top1_meter.avg:.2f}%, top5: {global_top5_meter.avg:.2f}%")
    # if run:
    #     run.log({f"top1_acc/mean": global_top1_meter.avg, "top5_acc/mean": global_top5_meter.avg})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset_path', type=str, default="/workspace/data/datasets/imagenet_c")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--wandb', type=str2bool, default=True)
    parser.add_argument('--wandb_resume', type=str2bool, default=False)

    # corruption parameters
    parser.add_argument('--corruptions', type=str, default=None)
    parser.add_argument('--severities', type=str, default=None)

    args = parser.parse_args()

    main(args)