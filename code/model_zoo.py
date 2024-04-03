import timm
import os
import logging
import torch
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


class LambdaModule(torch.nn.Module):

    def __init__(self, func):
        super(LambdaModule, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


NO_MEAN = [0, 0, 0]
NO_STD = [1, 1, 1]

URL_LOOKUP = {
    "resnet50_trained_on_SIN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar",
    "resnet50_trained_on_SIN_and_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
    "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
    # PRIME: A few primitives can boost robustness to common corruptions
    "resnet50_prime": "https://zenodo.org/record/5801872/files/ResNet50_ImageNet_PRIME_noJSD.ckpt?download=1",
    "resnet50_moco_v3_100ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/linear-100ep.pth.tar",
    "resnet50_moco_v3_300ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/linear-300ep.pth.tar",
    "resnet50_moco_v3_1000ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar",
}

GID_LOOKUP = {
    "resnet50_pixmix_90ep": "1_i45yvC88hos50QjkoD97OgbDGreKdp9",
    "resnet50_pixmix_180ep": "1cgKYXDym3wgquf-4hwr_Ra3qje6GNHPH",
    "resnet50_augmix_180ep": "1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF",
    "resnet50_deepaugment": "1DPRElQnBG66nd7GUphVm1t-5NroL7t7k",
    "resnet50_deepaugment_augmix": "1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP",
    "resnet50_noisymix": "1Na79fzPZ0Azg01h6kGn1Xu5NoWOElSuG",
    # https://arxiv.org/abs/2010.05981: "Shape-Texture Debiased Neural Network Training"
    "resnet50_tsbias_tbias": "1tFy2Q28rAqqreaS-27jifpyPzlohKDPH",
    "resnet50_tsbias_sbias": "1p0fZ9rU-J1v943tA7PlNMLuDXk_1Iy3K",
    "resnet50_tsbias_debiased": "1r-OSfTtrlFhB9GPiQXUE1h6cqosaBePw",
    "resnet50_frozen_random": "1IT65JJauql-Jdw-0AjAhEGGApGMVx-VE",
    # Patrick Müller
    "resnet50_opticsaugment": "1y0MSlVfzZBKZQEiZF2FCOmsk-91miIHR",
    # SimCLRv2
    "resnet50_simclrv2": "1fDaMhujPnxo4SYOe4asPqCKQ3XB_sZvj",
}


def r50_tf_to_torch(state):
    torch_state = {}
    for k, v in state.items():

        if "blocks" not in k:
            new_key = (
                k.replace("net.0.0.", "conv1.")
                .replace("net.0.1.0.", "bn1.")
                .replace("net0.0.", "conv1.")
                .replace("net.0.1.0.", "bn1.")
            )
        else:
            s = k.split(".")
            new_key = (
                "layer"
                + s[1]
                + "."
                + s[3]
                + "."
                + (k.replace(f"net.{s[1]}.blocks.{s[3]}.", ""))
            )
            new_key = (
                new_key.replace(".net.0", ".conv1")
                .replace(".net.1.0", ".bn1")
                .replace(".net.2", ".conv2")
                .replace(".net.3.0", ".bn2")
                .replace(".net.4", ".conv3")
                .replace(".net.5.0", ".bn3")
                .replace("projection.shortcut", "downsample.0")
                .replace("projection.bn.0", "downsample.1")
            )

        torch_state[new_key] = v

    return torch_state


def load_state_dict_from_gdrive(id, model_name, force_download=False):
    state_path = os.path.join(torch.hub.get_dir(), "checkpoints", f"{model_name}.pth")

    if not os.path.exists(state_path) or force_download:
        import gdown

        logging.info(f"Downloading {id} to {state_path}")
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        gdown.download(id=id, output=state_path, quiet=False)
        # download_file_from_google_drive(id, state_path)
    state = torch.load(state_path, map_location="cpu")
    return state


class NormalizedModel(torch.nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(
            torch.Tensor(mean).view(-1, 1, 1), requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.Tensor(std).view(-1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        out = (x - self.mean) / self.std
        out = self.model(out)
        return out


def get_normalized_model(model_name, eval=True):
    model = None

    # Good ol' AlexNet
    if model_name == "alexnet":
        model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # AT ResNets from Microsoft
    elif model_name.startswith("robust_resnet50"):
        model = timm.create_model("resnet50", pretrained=False)
        # get torch state from url
        tag = model_name.replace("robust_", "")
        state = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/{tag}.ckpt",
            map_location="cpu",
        )["model"]
        state = {k.replace("module.model.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # DINOv1 -> requires handling of seperated backbone/head
    elif model_name == "resnet50_dino":
        model = timm.create_model("resnet50", pretrained=False)
        # load backbone
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state, strict=False)
        # load classifier head
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth",
            map_location="cpu",
        )["state_dict"]
        state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # SwAV -> requires handling of seperated backbone/head
    elif model_name == "resnet50_swav":
        model = timm.create_model("resnet50", pretrained=False)
        # load backbone
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
        )
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        # load classifier head
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar",
            map_location="cpu",
        )["state_dict"]
        state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # Models accessible via http
    elif model_name in URL_LOOKUP:
        url = URL_LOOKUP.get(model_name)
        model = timm.create_model("resnet50", pretrained=False)
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # Improved torchvision R50
    elif model_name == "tv2_resnet50":
        model = torch.hub.load(
            "pytorch/vision", "resnet50", weights="ResNet50_Weights.IMAGENET1K_V2"
        )
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # Shape/texture bias regularized models -> require AuxBN handling
    elif model_name.startswith("resnet50_tsbias"):
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)["state_dict"]
        state = {
            k.replace("module.", ""): v for k, v in state.items() if "aux_bn" not in k
        }
        model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # DiffusionNoise model w/o noise frontend
    elif model_name == "resnet50_diffusionnoise_nonoise":
        from diffusion_noise.diffusionmodel import DiffusionNoiseModel

        url = "https://huggingface.co/paulgavrikov/resnet50.in1k_diffusionnoise_90ep/resolve/main/resnet50_diffusionnoise_nonorm_last.pth"

        model = timm.create_model("resnet50", pretrained=False)
        state = torch.hub.load_state_dict_from_url(url)["model"]
        model = DiffusionNoiseModel(model)
        model.load_state_dict(state)
        # Note: we pass model.module here!
        model = NormalizedModel(model.module, NO_MEAN, NO_STD)  # no normalization

    # SupCon
    elif model_name == "resnet50_supcon":
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)
        state = state["model"]
        state = {
            k.replace("module.", "")
            .replace("encoder.", "")
            .replace("head.2.", "fc."): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # SimCLRv2
    elif model_name == "resnet50_simclrv2":
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)
        model.load_state_dict(r50_tf_to_torch(state["resnet"]))
        model = NormalizedModel(model, NO_MEAN, NO_STD)  # no normalization

    # Models hosted on Google Drive
    elif model_name in GID_LOOKUP:
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)
        if "state_dict" in state.keys():
            state = state["state_dict"]
        if "model_state_dict" in state.keys():
            state = state["model_state_dict"]
        if "online_backbone" in state.keys():
            state = state["online_backbone"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # V1Net models
    elif model_name.startswith("vonenet_"):
        import vonenet

        tag = model_name.replace("vonenet_", "")
        model = vonenet.get_model(model_arch=tag, pretrained=True).module
        model = NormalizedModel(model, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)

    # Local Models
    elif model_name.startswith("file://"):
        path = model_name.replace("file://", "")
        model = timm.create_model("resnet50", pretrained=False)
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state.keys():
            state = state["state_dict"]
        if "model_state_dict" in state.keys():
            state = state["model_state_dict"]
        if "online_backbone" in state.keys():
            state = state["online_backbone"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # TorchVision ViT with DeiT-like training schedule w/o non-in1k data
    elif model_name == "vit_base_patch16_224.torchvision_deit":
        model = torch.hub.load(
            "pytorch/vision", "vit_b_16", weights="ViT_B_16_Weights.IMAGENET1K_V1"
        )
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # DINO ViT
    elif model_name == "vit_base_patch16_224.dino_in1k":

        state = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        fc_state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/paulgavrikov/in1k_head_for_vit_base_patch16_224.dino/resolve/main/checkpoint.pth.tar"
        )
        fc_state = {
            k.replace("module.linear.", ""): v
            for k, v in fc_state["state_dict"].items()
        }

        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        model.load_state_dict(
            state, strict=False
        )  # no strict because of the added head
        model.head = torch.nn.Linear(in_features=768, out_features=1000, bias=True)
        model.head.load_state_dict(fc_state)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # MAE ViT
    elif model_name == "vit_base_patch16_224.mae_in1k":
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, global_pool="avg"
        )
        state = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth"
        )["model"]
        model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # Default to timm models
    else:
        pretrained = True
        if model_name.endswith("_untrained"):
            pretrained = False
            model_name = model_name.replace("_untrained", "")
        model = timm.create_model(model_name, pretrained=pretrained)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    if eval:
        model.eval()
    return model
