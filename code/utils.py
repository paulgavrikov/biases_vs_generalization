import torch
from torch.utils.data import DataLoader
import torchvision
from collections import Counter
from model_zoo import get_normalized_model



TEST_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)


def get_imagenet_loader(path, batch_size, num_workers, shuffle=False):
    dataset = torchvision.datasets.ImageNet(
        path, split="val", transform=TEST_TRANSFORMS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def get_imagenet_folder_loader(
    path, batch_size, num_workers, shuffle=False, resize_crop=True
):
    transforms = TEST_TRANSFORMS if resize_crop else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.ImageFolder(path, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L420
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    torch.manual_seed(seed)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")


def get_gpu_stats():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

    nvmlInit()
    stats = []
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        stats.append(info.used)
    return stats


def autoselect_device():
    best_is_gpu = False

    device = "cpu"

    try:
        mps_available = torch.backends.mps.is_available()
    except:
        mps_available = False

    if torch.cuda.is_available():
        best_is_gpu = True
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"

    if best_is_gpu:
        import numpy as np

        best_device = f"cuda:{np.argmin(get_gpu_stats())}"
        device = best_device

    return device


def parse_aa_log(log_file):
    results = {}
    prev_attack = ""
    with open(log_file, "r") as file:
        for line in file.readlines():
            if "accuracy" in line:
                acc = (
                    float(line.split(": ")[1].replace("%", "").strip().split(" ")[0])
                    / 100
                )

                tag = None
                if "initial accuracy" in line:
                    tag = "clean"
                elif "after" in line:
                    tag = line.split(":")[0].split(" ")[-1].strip()
                    if len(prev_attack) == 0:
                        prev_attack = tag
                    else:
                        prev_attack += "+" + tag

                    tag = "AA-" + prev_attack
                else:
                    tag = "AA-robust"

                results[tag] = acc

    return results


def assert_imagenet_consistency(
    path, targets=1000, samples_per_class=50, total_samples=50000
):
    dataset = torchvision.datasets.ImageFolder(root=path, transform=None)

    assert (
        len(dataset.class_to_idx) == targets
    ), f"Expected {targets} targets, got {len(dataset.class_to_idx)}"
    assert (
        len(set(dataset.targets)) == targets
    ), f"Expected {targets} targets, got {len(set(dataset.targets))}"

    if samples_per_class is not None:
        for category, frequency in Counter(dataset.targets).most_common():
            target_name = get_key_for_value(dataset.class_to_idx, category)
            assert (
                frequency == samples_per_class
            ), f"Category {category} ({target_name}) has {frequency} samples, expected {samples_per_class}"

    assert (
        len(dataset) == total_samples
    ), f"Expected {total_samples} samples, got {len(dataset)}"


def get_key_for_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None


def tensor_map(func, iterable):
    return torch.stack([func(x) for x in iterable])
