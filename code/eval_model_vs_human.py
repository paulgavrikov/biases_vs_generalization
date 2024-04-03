import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, str2bool
import pandas as pd
from modelvshuman.utils import load_dataset
from modelvshuman.constants import DEFAULT_DATASETS
from modelvshuman.evaluation import evaluate as e
from modelvshuman.plotting.analyses import ShapeBias
from modelvshuman.plotting.plot import METRICS, get_mean_over_datasets
import copy
from modelvshuman import constants as c
from modelvshuman.datasets.experiments import get_experiments
from modelvshuman import constants as c
from modelvshuman.plotting.decision_makers import DecisionMaker
from functools import partial

try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


METRIC_RENAMER = {
    "accuracy (top-1)": "top1_acc",
    "accuracy (top-5)": "top5_acc",
}

def get_ood_mean(model_name):
    """
    This is a super hacky way to get the OOD accuracy ...
    """

    def plotting_definition_template(df, model_name):
        decision_makers = [DecisionMaker(name_pattern=model_name, df=df, plotting_name=model_name)]
        return decision_makers

    colname = "OOD accuracy"
    metric_fun, metric_name = METRICS[colname]
    datasets = get_experiments(c.DEFAULT_DATASETS)
    assert metric_name == "16-class-accuracy"
    df1 = get_mean_over_datasets(colname=colname,
                                    metric_fun=metric_fun,
                                    metric_name=metric_name,
                                    datasets=copy.deepcopy(datasets),
                                    decision_maker_fun=partial(plotting_definition_template, model_name=model_name))

    return df1[colname].item() * 100

def main(args):

    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="imagenet_model_vs_human", name=args.model, config=args)

    device = args.device

    d_mean = torch.nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), requires_grad=False)
    d_std = torch.nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), requires_grad=False)
    
    cue_conflict_csv = None

    for dataset_name in DEFAULT_DATASETS:
        dataset = load_dataset(dataset_name)

        model = get_normalized_model(args.model)
        model.to(device)

        for metric in dataset.metrics:
            metric.reset()

        with torch.no_grad():

            result_writer = e.ResultPrinter(model_name=args.model,
                                            dataset=dataset)

            for x, batch_targets, paths in tqdm(dataset.loader, desc=dataset_name):
                x = (x * d_std + d_mean).to(device)
                probs = torch.nn.functional.softmax(model(x), dim=1)
                preds = dataset.decision_mapping(probs.detach().cpu().numpy())

                for metric in dataset.metrics:
                    metric.update(preds, batch_targets, paths)
                        
                result_writer.print_batch_to_csv(object_response=preds,
                                                 batch_targets=batch_targets,
                                                 paths=paths)
            
            if dataset_name == "cue-conflict":
                cue_conflict_csv = result_writer.csv_file_path
        for metric in dataset.metrics:
            metric_name = METRIC_RENAMER.get(metric.name, metric.name) + "/" + dataset_name
            print(metric_name, metric.value)
            if run:
                run.log({metric_name: metric.value * 100})

    if cue_conflict_csv:
        df_cueconflict = pd.read_csv(cue_conflict_csv)
        shapebias_result = ShapeBias().analysis(df_cueconflict)
        print(shapebias_result)
        if run:
            run.log(shapebias_result)

    ood_mean = get_ood_mean(args.model)
    print("OOD accuracy", ood_mean)
    if run:
        run.log({"top1_acc/ood_mean": ood_mean})
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', type=str2bool, default=True)

    args = parser.parse_args()

    main(args)
