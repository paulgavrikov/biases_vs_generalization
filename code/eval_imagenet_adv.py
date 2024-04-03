import torch
import torch.utils.data
from tqdm import tqdm
import argparse
import os
from utils import get_normalized_model, get_imagenet_loader, str2bool, seed_everything, assert_imagenet_consistency
import foolbox as fb
import os
from autoattack import AutoAttack
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


def parse_aa_log(log_file):
    results = {}
    prev_attack = ""
    with open(log_file, "r") as file:
        for line in file.readlines():
            if "accuracy" in line:
                acc = float(line.split(": ")[1].replace("%", "").strip().split(" ")[0]) / 100

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


def run_foolbox_attack(attack, model, x, y, eps, batch_size, device):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    pos = 0
    perturbed = 0
    progress = tqdm(total=len(x), desc=f"Running {attack.__class__.__name__}")
    while pos < len(x):
        _, _, success = attack(fmodel, x[pos:pos + batch_size].to(device), y[pos:pos + batch_size].to(device), epsilons=[eps])
        perturbed += success.float().sum(axis=-1)[0].item()
        pos += batch_size
        progress.update(batch_size)
        progress.set_postfix({"acc": 1 - (perturbed / pos)})

    return 1 - (perturbed / pos)


def deque_loader(loader, n_samples=-1):
    all_x = []
    all_y = []

    total = len(loader) if n_samples == -1 else n_samples

    for x, y in tqdm(loader, total=total, desc="Loading data"):
        all_x.append(x)
        all_y.append(y)

        if n_samples != -1:
            break

    all_x = torch.vstack(all_x)
    all_y = torch.hstack(all_y)
    return all_x, all_y


def main(args):

    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="imagenet_adv", name=args.model, config=args)

    seed_everything(args.seed)

    device = args.device
    
    assert_imagenet_consistency(os.path.join(args.imagenet, "val"))
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=16, shuffle=False)
    all_x, all_y = deque_loader(dataloader, n_samples=args.n_samples)

    model = get_normalized_model(args.model)
    model.to(device)

    if args.pgd:
        pgd_steps = 40
        acc = run_foolbox_attack(fb.attacks.LinfPGD(steps=pgd_steps, abs_stepsize=2/255), model, all_x, all_y, args.eps, args.batch_size, args.device)
        print(f"PGD: {acc}")
        if run is not None:
            run.log({f"PGD-{pgd_steps}": acc})

    if args.fgsm:
        acc = run_foolbox_attack(fb.attacks.FGSM(), model, all_x, all_y, args.eps, args.batch_size, args.device)
        print(f"FGSM: {acc}")
        if run is not None:
            run.log({"FGSM": acc})

    if args.aa:

        log_file = f"autoattack/{args.model}_aa_log.txt"

        if os.path.isfile(log_file):
            os.remove(log_file)

        adversary = AutoAttack(model, norm="Linf", eps=args.eps, log_path=log_file, device=args.device, version="standard")
        _ = adversary.run_standard_evaluation(all_x, all_y, bs=args.batch_size)
        for k, v in parse_aa_log(log_file).items():
            print(f"{k}: {v}")
            if run is not None:
                run.log({k: v})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--imagenet', type=str, default="/workspace/data/datasets/imagenet/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=str2bool, default=True)

    # attack parameter
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--fgsm', type=str2bool, default=False)
    parser.add_argument('--pgd', type=str2bool, default=True)
    parser.add_argument('--aa', type=str2bool, default=False)
    parser.add_argument('--eps', type=float, default=0.5/255)

    args = parser.parse_args()

    main(args)
