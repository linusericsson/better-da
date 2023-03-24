import os
import json
import copy
import argparse
from tkinter import W

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import torch.nn.functional as F

from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler

from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.datasets import DataloaderCreator, get_mnist, get_office31, get_officehome, get_visda2017c
from pytorch_adapt.hooks import ATDOCHook, BNMHook, BSPHook, CDANHook, ClusteringHook, DANNHook, GVBHook, MCDHook, MCCHook, AlignerPlusCHook, ClassifierHook, FinetunerHook, TargetDiversityHook, TargetEntropyHook
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.models import mnistG, mnistC, resnet18, resnet50, resnet101, Classifier, Discriminator
from pytorch_adapt.validators import AccuracyValidator, EntropyValidator, BNMValidator, IMValidator, SNDValidator, DeepEmbeddedValidator, HenryValidator
from pytorch_adapt.layers import RandomizedDotProduct, MultipleModels, SlicedWasserstein, ModelWithBridge, MCCLoss, MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales
from pytorch_adapt.weighters import MeanWeighter


# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="office31")
parser.add_argument('--source', type=str, default="dslr")
parser.add_argument('--target', type=str, default="amazon")

parser.add_argument('--algorithm', type=str, default="source-only")
parser.add_argument('--G-arch', type=str, default="resnet50")
parser.add_argument('--init-source-only', action="store_true", default=False)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=64)

parser.add_argument('--hpo', type=str, default="random")
parser.add_argument('--hpo-validator', type=str, default="src_val_acc_score")
parser.add_argument('--hpo-validate-freq', type=int, default=10)
parser.add_argument('--hpo-max-epochs', type=int, default=200)
parser.add_argument('--hpo-early-stopping-patience', type=int, default=200)
parser.add_argument('--hpo-gpus-per-trial', type=float, default=1)
parser.add_argument('--hpo-cpus-per-trial', type=int, default=8)
parser.add_argument('--hpo-num-samples', type=int, default=10)

#parser.add_argument('--epochs', type=int, default=15)
#parser.add_argument('--lr', type=float, default=2e-4)
#parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--feature-layer', type=str, default="fl3")
parser.add_argument('--validators', nargs='+', type=str,
    default=[
        "src_train_acc_score",
        "src_val_acc_score",
        "target_train_acc_score",
        "target_val_acc_score",
        "target_test_acc_score",
    ])

parser.add_argument('--device', type=str, default="cuda")

parser.add_argument('--results-root', type=str, default="/home/CORP/l.ericsson/code/pytorch-adapt/examples/visda/results/")
parser.add_argument('--data-root', type=str, default="/home/CORP/l.ericsson/data/")

parser.add_argument('--debug', action='store_true', default=False)


ARCHITECTURES = {
    "mnistG": mnistG,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101
}
DATASETS = {
    "mnistm": (get_mnist, 10),
    "visda2017c": (get_visda2017c, 12),
    "office31": (get_office31, 31),
    "officehome": (get_officehome, 65),
}
HOOKS = {
    "source-only": lambda kwargs: FinetunerHook(
        opts=[kwargs["opts"][1]]
    ),
    "atdoc":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[ATDOCHook(
            dataset_size=kwargs["dataset_size"],
            feature_dim=kwargs["feature_dim"],
            num_classes=kwargs["num_classes"],
            k=kwargs["config"]["k_atdoc"]
        )],
        weighter=MeanWeighter(weights={
            "pseudo_label_loss": kwargs["config"]["lambda_atdoc"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "bnm":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[BNMHook()],
        weighter=MeanWeighter(weights={
            "bnm_loss": kwargs["config"]["lambda_bnm"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "bsp":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[BSPHook()],
        weighter=MeanWeighter(weights={
            "bsp_loss": kwargs["config"]["lambda_bsp"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "cdan": lambda kwargs: CDANHook(
        g_opts=[kwargs["opts"][0], kwargs["opts"][1]], d_opts=[kwargs["opts"][2]],
        d_weighter=MeanWeighter(scale=kwargs["config"]["lambda_D"]),
        g_weighter=MeanWeighter(weights={
            "src_domain_loss": kwargs["config"]["lambda_G"],
            "target_domain_loss": kwargs["config"]["lambda_G"],
            "c_loss": kwargs["config"]["lambda_L"],
        }),
    ),
    "clustering":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[ClusteringHook(
            num_classes=kwargs["num_classes"],
            with_src=kwargs["config"]["lambda_src"],
            centroid_init=kwargs["config"]["lambda_c_init"],
            feat_normalize=kwargs["config"]["lambda_norm"],
        )],
        weighter=MeanWeighter(weights={
            "pseudo_label_loss": kwargs["config"]["lambda_clustering"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "dann": lambda kwargs: DANNHook(
        opts=kwargs["opts"][:3],
        weighter=MeanWeighter(weights={
            "src_domain_loss": kwargs["config"]["lambda_D"],
            "target_domain_loss": kwargs["config"]["lambda_D"],
            "c_loss": kwargs["config"]["lambda_L"],
        }),
        gradient_reversal_weight=kwargs["config"]["lambda_grl"]
    ),
    "gvb":  lambda kwargs: GVBHook(
        opts=kwargs["opts"][:3],
        weighter=MeanWeighter(weights={
            "src_domain_loss": kwargs["config"]["lambda_D"],
            "target_domain_loss": kwargs["config"]["lambda_D"],
            "g_src_bridge_loss": kwargs["config"]["lambda_B_G"],
            "d_src_bridge_loss": kwargs["config"]["lambda_B_D"],
            "g_target_bridge_loss": kwargs["config"]["lambda_B_G"],
            "d_target_bridge_loss": kwargs["config"]["lambda_B_D"],
        }),
        gradient_reversal_weight=kwargs["config"]["lambda_grl"]
    ),
    "im":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[TargetEntropyHook(), TargetDiversityHook()],
        weighter=MeanWeighter(weights={
            "entropy_loss": kwargs["config"]["lambda_imax"],
            "diversity_loss": kwargs["config"]["lambda_imax"],
            "c_loss": kwargs["config"]["lambda_L"],
        })

    ),
    "mcc":  lambda kwargs: ClassifierHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        post=[MCCHook(loss_fn=MCCLoss(T=kwargs["config"]["T_mcc"]))],
        weighter=MeanWeighter(weights={
            "mcc_loss": kwargs["config"]["lambda_mcc"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "mcd":  lambda kwargs: MCDHook(
        g_opts=[kwargs["opts"][0]],
        c_opts=[kwargs["opts"][1]],
        repeat=kwargs["config"]["N_mcd"],
        x_weighter=MeanWeighter(scale=kwargs["config"]["lambda_L"]),
        y_weighter=MeanWeighter(weights={
            "c_loss0": kwargs["config"]["lambda_L"],
            "c_loss1": kwargs["config"]["lambda_L"],
            "discrepancy_loss": kwargs["config"]["lambda_disc"],
        }),
        z_weighter=MeanWeighter(scale=kwargs["config"]["lambda_disc"]),
    ),
    "mmd":  lambda kwargs: AlignerPlusCHook(
        opts=[kwargs["opts"][0], kwargs["opts"][1]],
        loss_fn=MMDLoss(kernel_scales=get_kernel_scales(
            low=-kwargs["config"]["gamma_exp"],
            high=kwargs["config"]["gamma_exp"],
            num_kernels=(kwargs["config"]["gamma_exp"] * 2) + 1
        )),
        softmax=True,
        weighter=MeanWeighter(weights={
            "features_confusion_loss": kwargs["config"]["lambda_F"],
            "logits_confusion_loss": kwargs["config"]["lambda_F"],
            "c_loss": kwargs["config"]["lambda_L"],
        })
    ),
    "swd":  lambda kwargs: MCDHook(
        g_opts=[kwargs["opts"][0]],
        c_opts=[kwargs["opts"][1]],
        discrepancy_loss_fn=SlicedWasserstein(m=128),
        repeat=kwargs["config"]["generator_updates"]
    ),
}
VALIDATORS = {
    "src_train_acc_score": lambda kwargs: (AccuracyValidator(), {"src_val": "src_train"}),
    "src_val_acc_score": lambda kwargs: (AccuracyValidator(), {"src_val": "src_val"}),
    "target_train_acc_score": lambda kwargs: (AccuracyValidator(), {"src_val": "target_train"}),
    "target_val_acc_score": lambda kwargs: (AccuracyValidator(), {"src_val": "target_val"}),
    "target_test_acc_score": lambda kwargs: (AccuracyValidator(), {"src_val": "target_test"}),

    "src_train_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "src_train"}),
    "src_val_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "src_val"}),
    "target_train_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "target_train"}),
    "target_val_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "target_val"}),

    "src_train_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "src_train"}),
    "src_val_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "src_val"}),
    "target_train_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "target_train"}),
    "target_val_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "target_val"}),

    "src_train_im_score": lambda kwargs: (IMValidator(), {"target_train": "src_train"}),
    "src_val_im_score": lambda kwargs: (IMValidator(), {"target_train": "src_val"}),
    "target_train_im_score": lambda kwargs: (IMValidator(), {"target_train": "target_train"}),
    "target_val_im_score": lambda kwargs: (IMValidator(), {"target_train": "target_val"}),

    "src_train_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "src_train"}),
    "src_val_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "src_val"}),
    "target_train_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "target_train"}),
    "target_val_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "target_val"}),

    "target_train_henry_score": lambda kwargs: (HenryValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val", "target_train": "target_train"}),
    "target_val_henry_score": lambda kwargs: (HenryValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val", "target_train": "target_val"}),

    "dev_features_max_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="features",
            normalization="max"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
    "dev_logits_max_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="logits",
            normalization="max"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
    "dev_preds_max_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="preds",
            normalization="max"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
    "dev_features_standardize_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="features",
            normalization="standardize"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
    "dev_logits_standardize_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="logits",
            normalization="standardize"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
    "dev_preds_standardize_score": lambda kwargs: (
        DeepEmbeddedValidator(
            temp_folder=".dev_temp",
            layer="preds",
            normalization="standardize"
        ),
        {"src_train": "src_train", "src_val": "src_val", "target_train": "target_train"}
    ),
}


class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, start_multiplier, warmup_epochs):
        self.start_multiplier = start_multiplier
        self.warmup_epochs = warmup_epochs
        self.warmup_multipliers = torch.linspace(self.start_multiplier, 1.0, self.warmup_epochs + 1)
        self.after_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [base_lr * self.warmup_multipliers[self.last_epoch] for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(CosineAnnealingWarmupLR, self).step(epoch)


class Validator():
    def __init__(self, validators, main_validator, patience):
        self.validators = validators
        self.main_validator = main_validator
        self.patience = patience
        self.scores = []

    def compute(self, data, epoch):
        d = {"epoch": epoch + 1}
        for name, (validator, input_keys) in self.validators.items():
            try:
                d[name] = validator(**{key: data[value] for key, value in input_keys.items()})
            except Exception as e: print(e)
        self.scores.append(d)

    def necessary_splits(self):
        splits = list(set([val for v in self.validators for val in self.validators[v][1].values()]))
        split_splits = [key.split("_") for key in splits]
        splits_with_labels = []
        for split in split_splits:
            s = f"{split[1]}_with_labels" if split[0] == "target" else split[1]
            splits_with_labels.append([split[0], s])
        return splits, splits_with_labels

    def early_stopping(self):
        history = torch.tensor(self.get(self.main_validator, recent=False))
        best_val_step = torch.argmax(history)
        self.best_epoch = self.scores[best_val_step]["epoch"]
        current_val_step = len(self.scores) - 1
        return current_val_step - best_val_step > self.patience

    def get(self, name, recent=True):
        if recent:
            return self.scores[-1][name]
        else:
            return [self.scores[i][name] for i in range(len(self.scores))]

    def get_best(self, name, best_epoch):
        history = [s for s in self.scores]
        for h in history:
            if h["epoch"] == best_epoch:
                return h
        return None

    def print(self, recent=True):
        history = [self.scores[-1]] if recent else self.scores
        for scores in history:
            for k, v in scores.items():
                print(f"\t{k}:\t{v:.2f}")
        print("\n")

    def save(self, results_root, dataset, source, target, algorithm, filename, indent=4):
        with open(os.path.join(results_root, dataset, source, target, algorithm, filename), "w") as f:
            json.dump(self.scores[-1], f, indent=indent)


def inference(models, dataloaders, device, domain="src", split="train", feature_layer="fl0"):
    G, C = models["G"], models["C"]
    labels, fl0s, fl3s, fl6s, logits, preds = [], [], [], [], [], []
    with torch.no_grad():
        for data in tqdm(dataloaders[f"{domain}_{split}"], desc=f"{domain}_{split}"):
            data = c_f.batch_to_device(data, device)
            fl0 = G(data[f"{domain}_imgs"])
            logit, fl6, fl3 = C(fl0, return_all_features=True)
            if isinstance(logit, list):
                logit = logit[0]
            pred = F.softmax(logit, dim=-1)
            fl0s.append(fl0)
            fl3s.append(fl3)
            fl6s.append(fl6)
            logits.append(logit)
            preds.append(pred)
            if f"{domain}_labels" in data:
                label = data[f"{domain}_labels"]
                labels.append(label)
    fl0s = torch.cat(fl0s, dim=0)
    fl3s = torch.cat(fl3s, dim=0)
    fl6s = torch.cat(fl6s, dim=0)
    logits = torch.cat(logits, dim=0)
    preds = torch.cat(preds, dim=0)
    if labels:
        labels = torch.cat(labels, dim=0)
        data = {"labels": labels, "fl0": fl0s, "fl3": fl3s, "fl6": fl6s, "logits": logits, "preds": preds}
    else:
        data = {"fl0": fl0s, "fl3": fl3s, "fl6": fl6s, "logits": logits, "preds": preds}
    data["features"] = data[feature_layer]
    return data


def train(args, epoch, hook, models, optimizers, misc, dataloaders, data_key, device, save):
    models.train()
    for data in tqdm(dataloaders[data_key], desc=f"Epoch {epoch}"):
        data = c_f.batch_to_device(data, device)
        _, loss = hook({**models, **misc, **data})
        if args.debug:
            break


def evaluate(args, run_name, epoch, models, optimizers, misc, dataloaders, device, validator, save):
    print("Computing validation scores")
    models.eval()
    eval_data = {a: inference(models, dataloaders, device, *b, feature_layer=args.feature_layer) for a, b in zip(*validator.necessary_splits())}
    validator.compute(eval_data, epoch)
    validator.print()
    if save:
        # Save checkpoint
        checkpoint = {
            "args": args,
            "epoch": epoch + 1,
            "models": {m: models[m].module.state_dict() for m in models},
            "optimizers": [o.state_dict() for o in optimizers],
            "misc": [misc[m].state_dict() for m in misc]
        }
        if args.algorithm == "source-only":
            torch.save(checkpoint, os.path.join(args.results_root, args.dataset, args.source, args.target, args.algorithm, f"checkpoint_{run_name}_{epoch + 1}.pt"))
        torch.save(eval_data, os.path.join(args.results_root, args.dataset, args.source, args.target, args.algorithm, f"features_{run_name}_{epoch + 1}.pt"))
        validator.save(args.results_root, args.dataset, args.source, args.target, args.algorithm, f"scores_{run_name}_{epoch + 1}.json")
    return validator.early_stopping(), validator.best_epoch


def run(config):
    global args

    run_name = str(config)
    print(f"Running job with config: {config}")

    start_epoch = 0

    device = torch.device(args.device)

    # Create datasets and dataloaders
    if args.algorithm == "source-only":
        train_names, val_names = ["src_train"], ["train", "src_val", "target_train", "target_train_with_labels", "target_val", "target_val_with_labels", "target_test", "target_test_with_labels"]
    else:
        train_names, val_names = ["train"], ["src_train", "src_val", "target_train", "target_train_with_labels", "target_val", "target_val_with_labels", "target_test", "target_test_with_labels"]

    dataset_class, num_classes = DATASETS[args.dataset]
    datasets = dataset_class([args.source], [args.target], folder=args.data_root, return_target_with_labels=True, download=False)
    dc = DataloaderCreator(
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_names=train_names,
        val_names=val_names
    )
    dataloaders = dc(**datasets)
    target_dataset_size = len(datasets["target_train"])


    # ### Create models, optimizers, hook, and validator
    def get_G(args, device):
        G = ARCHITECTURES[args.G_arch](pretrained=True)
        feature_dim = {"mnistG": 1200, "resnet18": 512, "resnet50": 2048, "resnet101": 2048}[args.G_arch]
        if args.init_source_only:
            G_state_dict = torch.load(os.path.join(args.results_root, args.dataset, args.source, args.target, "source-only", "best.pt"))["models"]["G"]
            G.load_state_dict(G_state_dict, strict=True)
        return torch.nn.DataParallel(G.to(device)), feature_dim
    G, feature_dim = get_G(args, device)

    def get_C(args, num_classes, feature_dim, h_dim, device):
        C = Classifier(num_classes, feature_dim, h_dim)
        if args.init_source_only:
            C_state_dict = torch.load(os.path.join(args.results_root, args.dataset, args.source, args.target, "source-only", "best.pt"))["models"]["C"]
            C.load_state_dict(C_state_dict, strict=True)
            print("Source-only checkpoint loaded")
        if args.algorithm in ["mcd", "swd"]:
            C = MultipleModels(C, c_f.reinit(copy.deepcopy(C)))
        if args.algorithm == "gvb":
            C = ModelWithBridge(C)
        return torch.nn.DataParallel(C.to(device))
    C = get_C(args, num_classes, feature_dim, 256, device)

    def get_D(args, num_classes, feature_dim, device):
        if args.algorithm == "gvb":
            # Discriminator comes after classifier,
            # so the input shape is num_classes instead of feature size
            D_ = torch.nn.Sequential(torch.nn.Linear(num_classes, 1), torch.nn.Flatten(start_dim=0))
            D = ModelWithBridge(D).to(device)
        else:
            D = Discriminator(in_size=feature_dim)
        return torch.nn.DataParallel(D.to(device))
    D = get_D(args, num_classes, feature_dim, device)

    models = Models({"G": G, "C": C, "D": D})

    optimizers = Optimizers(
        (torch.optim.Adam, {"lr": config["lr"], "weight_decay": args.weight_decay}),
        multipliers={"G": 1., "C": 1., "D": 1.}
    )
    optimizers.create_with(models)
    schedulers = LRSchedulers((
        CosineAnnealingWarmupLR,
        {"T_max": args.hpo_max_epochs, "eta_min": 0, "start_multiplier": 0.01, "warmup_epochs": int(args.hpo_max_epochs * 0.05)}
    ))
    schedulers.create_with(optimizers)
    optimizers = list(optimizers.values())

    feature_combiner = RandomizedDotProduct(in_dims=[feature_dim, num_classes], out_dim=feature_dim)
    misc = {"feature_combiner": feature_combiner}

    hook = HOOKS[args.algorithm]({
        "opts": optimizers,
        "config": config,
        "dataset_size": target_dataset_size,
        "feature_dim": feature_dim,
        "num_classes": num_classes
    })

    chosen_validators = {key: VALIDATORS[key]({"num_classes": num_classes}) for key in args.validators}
    validator = Validator(chosen_validators, args.hpo_validator, args.hpo_early_stopping_patience)

    if "best_epoch" in config and args.hpo is None:
        max_epochs = config["best_epoch"]
    else:
        max_epochs = args.hpo_max_epochs
    #evaluate(args, run_name, -1, models, optimizers, misc, dataloaders, device, validator, save=False)
    for epoch in range(start_epoch, max_epochs):
        train(args, epoch, hook, models, optimizers, misc, dataloaders, data_key=train_names[0], device=device, save=args.hpo is None)
        schedulers.step("per_epoch")

        if not (epoch + 1) % args.hpo_validate_freq:
            early_stopping, best_epoch = evaluate(args, run_name, epoch, models, optimizers, misc, dataloaders, device, validator, save=args.hpo is not None)

            if args.hpo is not None:
                session.report({key: validator.get(key) for key in args.validators})

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == "office31":
        if args.target == "amazon":
            args.hpo_validate_freq = 10
            args.hpo_max_epochs = 200
            args.hpo_early_stopping_patience = 200
        else:
            args.hpo_validate_freq = 100
            args.hpo_max_epochs = 2000
            args.hpo_early_stopping_patience = 2000

    os.makedirs(os.path.join(args.results_root, args.dataset, args.source, args.target, args.algorithm), exist_ok=True)

    config = {
        "lr": tune.loguniform(1e-5, 1e-1)
    }
    if args.algorithm == "atdoc":
        config["lambda_atdoc"] = tune.uniform(0, 1)
        config["k_atdoc"] = tune.qrandint(5, 25, 5)
        config["lambda_L"] = tune.uniform(0, 1)
    elif args.algorithm == "bnm":
        config["lambda_bnm"] = tune.uniform(0, 1)
        config["lambda_L"] = tune.uniform(0, 1)
    if args.algorithm == "bsp":
        config["lambda_bsp"] = tune.loguniform(1e-6, 1)
        config["lambda_L"] = tune.uniform(0, 1)
    elif args.algorithm == "cdan":
        config["lambda_D"] = tune.uniform(0, 1)
        config["lambda_G"] = tune.uniform(0, 1)
        config["lambda_L"] = tune.uniform(0, 1)
    elif args.algorithm == "clustering":
        config["lambda_clustering"] = tune.uniform(0, 1)
        config["lambda_L"] = tune.uniform(0, 1)
        config["lambda_src"] = tune.choice([False, True])
        config["lambda_c_init"] = tune.choice([False, True])
        config["lambda_norm"] = tune.choice([False, True])
    elif args.algorithm == "dann":
        config["lambda_D"] = tune.uniform(0, 1)
        config["lambda_grl"] = tune.loguniform(0.1, 10)
        config["lambda_L"] = tune.uniform(0, 1)
    elif args.algorithm == "gvb":
        config["lambda_D"] = tune.uniform(0, 1)
        config["lambda_B_G"] = tune.uniform(0, 1)
        config["lambda_B_D"] = tune.uniform(0, 1)
        config["lambda_grl"] = tune.loguniform(0.1, 10)
    elif args.algorithm == "mcc":
        config["lambda_mcc"] = tune.uniform(0, 1)
        config["T_mcc"] = tune.uniform(0.2, 5)
        config["lambda_L"] = tune.uniform(0, 1)
    elif args.algorithm in ["mcd"]:
        config["N_mcd"] = tune.randint(1, 11)
        config["lambda_L"] = tune.uniform(0, 1)
        config["lambda_disc"] = tune.uniform(0, 1)
    elif args.algorithm == "mmd":
        config["lambda_F"] = tune.uniform(0, 1)
        config["lambda_L"] = tune.uniform(0, 1)
        config["gamma_exp"] = tune.randint(1, 9)
    elif args.algorithm in ["mcd", "swd"]:
        config["generator_updates"] = tune.choice([1, 2, 3, 4, 5])

    if args.hpo == "asha":
        scheduler = ASHAScheduler(
            max_t=args.hpo_max_epochs,
            grace_period=1,
            reduction_factor=2
        )
    elif args.hpo == "random":
        scheduler = FIFOScheduler()
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources={"cpu": args.hpo_cpus_per_trial, "gpu": args.hpo_gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric=args.hpo_validator,
            mode="max",
            scheduler=scheduler,
            num_samples=args.hpo_num_samples,
        ),
        run_config=air.RunConfig(local_dir=os.path.join(args.results_root, "ray_results"), failure_config=air.FailureConfig(fail_fast=True)),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result(args.hpo_validator, "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation scores:")
    for k in args.validators:
        print(f"\t{k}:\t{best_result.metrics[k]:.2f}")
