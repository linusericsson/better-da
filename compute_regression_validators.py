import os, glob
import json
import copy
import argparse
from tkinter import W

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    fowlkes_mallows_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    accuracy_score, f1_score, log_loss
)

from copy import deepcopy

import torch
import torch.nn.functional as F

from pytorch_adapt.datasets import get_mnistmr, get_dogs_and_birds
from pytorch_adapt.validators import (
    MSEValidator, MAEValidator,
    EntropyValidator, EntropyCombinedValidator, IMValidator, IMCombinedValidator,
    BNMValidator, BNMWithSourceValidator,
    ClassClusterValidator, PerClassValidator,
    SNDValidator, MMDValidator,
    HenryValidator, HenryV3Validator, HenryV3RegressionValidator, HenryUpdatedValidator,
    GoukValidator, GoukV3Validator, GoukUpdatedValidator, ImprovedGoukValidator,
    CORALValidator, RankMeValidator
)

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnistmr")
parser.add_argument('--source', type=str, default="mnist")
parser.add_argument('--target', type=str, default="mnistm")

parser.add_argument('--algorithm', type=str, default="source-only")
parser.add_argument('--G-arch', type=str, default="mnistG")
parser.add_argument('--init-source-only', action="store_true", default=False)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=64)

parser.add_argument('--feature-layer', type=str, default="fl3")
parser.add_argument('--validators', nargs='+', type=str,
    default=[
        "src_train_mse_score",
        "src_val_mse_score",
        "target_train_mse_score",
        "target_val_mse_score",
        "target_test_mse_score",

        "src_train_mae_score",
        "src_val_mae_score",
        "target_train_mae_score",
        "target_val_mae_score",
        "target_test_mae_score",

        "src_train_bnm_score",
        "src_val_bnm_score",
        "target_train_bnm_score",
        "target_val_bnm_score",

        "src_train_target_train_bnm_score",
        "src_train_target_val_bnm_score",
        "src_val_target_train_bnm_score",
        "src_val_target_val_bnm_score",

        "src_train_entropy_score",
        "src_val_entropy_score",
        "target_train_entropy_score",
        "target_val_entropy_score",

        "src_train_target_train_entropy_score",
        "src_train_target_val_entropy_score",
        "src_val_target_train_entropy_score",
        "src_val_target_val_entropy_score",

        "src_train_im_score",
        "src_val_im_score",
        "target_train_im_score",
        "target_val_im_score",

        "src_train_target_train_im_score",
        "src_train_target_val_im_score",
        "src_val_target_train_im_score",
        "src_val_target_val_im_score",

        "src_train_class_<score_fn_name>_score",
        "src_val_class_<score_fn_name>_score",
        "target_train_class_<score_fn_name>_score",
        "target_val_class_<score_fn_name>_score",

        "src_train_target_train_class_<score_fn_name>_score",
        "src_train_target_val_class_<score_fn_name>_score",
        "src_val_target_train_class_<score_fn_name>_score",
        "src_val_target_val_class_<score_fn_name>_score",

        "src_train_logits_class_<score_fn_name>_score",
        "src_val_logits_class_<score_fn_name>_score",
        "target_train_logits_class_<score_fn_name>_score",
        "target_val_logits_class_<score_fn_name>_score",

        "src_train_target_train_logits_class_<score_fn_name>_score",
        "src_train_target_val_logits_class_<score_fn_name>_score",
        "src_val_target_train_logits_class_<score_fn_name>_score",
        "src_val_target_val_logits_class_<score_fn_name>_score",

        "src_train_snd_score",
        "src_val_snd_score",
        "target_train_snd_score",
        "target_val_snd_score",

        "src_train_target_train_mmd_score",
        "src_train_target_val_mmd_score",
        "src_val_target_train_mmd_score",
        "src_val_target_val_mmd_score",

        "src_train_target_train_logits_mmd_score",
        "src_train_target_val_logits_mmd_score",
        "src_val_target_train_logits_mmd_score",
        "src_val_target_val_logits_mmd_score",

        "src_train_target_train_preds_mmd_score",
        "src_train_target_val_preds_mmd_score",
        "src_val_target_train_preds_mmd_score",
        "src_val_target_val_preds_mmd_score",

        "src_train_target_train_mmd_per_class_score",
        "src_train_target_val_mmd_per_class_score",
        "src_val_target_train_mmd_per_class_score",
        "src_val_target_val_mmd_per_class_score",

        "src_train_target_train_logits_mmd_per_class_score",
        "src_train_target_val_logits_mmd_per_class_score",
        "src_val_target_train_logits_mmd_per_class_score",
        "src_val_target_val_logits_mmd_per_class_score",

        "src_train_target_train_preds_mmd_per_class_score",
        "src_train_target_val_preds_mmd_per_class_score",
        "src_val_target_train_preds_mmd_per_class_score",
        "src_val_target_val_preds_mmd_per_class_score",

        "src_train_target_train_coral_score",
        "src_train_target_val_coral_score",
        "src_val_target_train_coral_score",
        "src_val_target_val_coral_score",

        "src_train_target_train_logits_coral_score",
        "src_train_target_val_logits_coral_score",
        "src_val_target_train_logits_coral_score",
        "src_val_target_val_logits_coral_score",

        "src_train_target_train_preds_coral_score",
        "src_train_target_val_preds_coral_score",
        "src_val_target_train_preds_coral_score",
        "src_val_target_val_preds_coral_score",

        "src_train_target_train_coral_per_class_score",
        "src_train_target_val_coral_per_class_score",
        "src_val_target_train_coral_per_class_score",
        "src_val_target_val_coral_per_class_score",

        "src_train_target_train_logits_coral_per_class_score",
        "src_train_target_val_logits_coral_per_class_score",
        "src_val_target_train_logits_coral_per_class_score",
        "src_val_target_val_logits_coral_per_class_score",

        "src_train_target_train_preds_coral_per_class_score",
        "src_train_target_val_preds_coral_per_class_score",
        "src_val_target_train_preds_coral_per_class_score",
        "src_val_target_val_preds_coral_per_class_score",

        "src_train_rank_me_score",
        "src_val_rank_me_score",
        "target_train_rank_me_score",
        "target_val_rank_me_score",

        "src_train_logits_rank_me_score",
        "src_val_logits_rank_me_score",
        "target_train_logits_rank_me_score",
        "target_val_logits_rank_me_score",

        "src_train_preds_rank_me_score",
        "src_val_preds_rank_me_score",
        "target_train_preds_rank_me_score",
        "target_val_preds_rank_me_score",

        "src_train_target_train_rank_me_score",
        "src_train_target_val_rank_me_score",
        "src_val_target_train_rank_me_score",
        "src_val_target_val_rank_me_score",

        "src_train_target_train_logits_rank_me_score",
        "src_train_target_val_logits_rank_me_score",
        "src_val_target_train_logits_rank_me_score",
        "src_val_target_val_logits_rank_me_score",

        "src_train_target_train_preds_rank_me_score",
        "src_train_target_val_preds_rank_me_score",
        "src_val_target_train_preds_rank_me_score",
        "src_val_target_val_preds_rank_me_score",

        # "target_train_henry_score",
        # "target_val_henry_score",

        # "src_train_gouk_score",
        # "src_val_gouk_score",

        # "target_train_henry_v3_score",
        # "target_val_henry_v3_score",

        # "src_train_gouk_v3_score",
        # "src_val_gouk_v3_score",

        # "target_train_henry_updated_score",
        # "target_val_henry_updated_score",

        # "src_train_gouk_updated_score",
        # "src_val_gouk_updated_score",

        # "src_train_improved_gouk_score",
        # "src_val_improved_gouk_score",
    ])

parser.add_argument('--device', type=str, default="cuda")

parser.add_argument('--results-root', type=str,
                    default="/home/CORP/l.ericsson/code/pytorch-adapt/examples/visda/results/")
parser.add_argument('--data-root', type=str, default="/home/CORP/l.ericsson/data/")

parser.add_argument('--debug', action='store_true', default=False)

DATASETS = {
    # the second value is num_classes, which in this setting is the number of classes in the discretized prediction space
    "mnistmr": (get_mnistmr, 8 ** 4, 4, 32),
    "dogs_and_birds": (get_dogs_and_birds, 8 ** 4, 4, 224),
}
VALIDATORS = {
    "src_train_mse_score": lambda kwargs: (MSEValidator(), {"src_val": "src_train"}),
    "src_val_mse_score": lambda kwargs: (MSEValidator(), {"src_val": "src_val"}),
    "target_train_mse_score": lambda kwargs: (MSEValidator(), {"src_val": "target_train"}),
    "target_val_mse_score": lambda kwargs: (MSEValidator(), {"src_val": "target_val"}),
    "target_test_mse_score": lambda kwargs: (MSEValidator(), {"src_val": "target_test"}),

    "src_train_mae_score": lambda kwargs: (MAEValidator(), {"src_val": "src_train"}),
    "src_val_mae_score": lambda kwargs: (MAEValidator(), {"src_val": "src_val"}),
    "target_train_mae_score": lambda kwargs: (MAEValidator(), {"src_val": "target_train"}),
    "target_val_mae_score": lambda kwargs: (MAEValidator(), {"src_val": "target_val"}),
    "target_test_mae_score": lambda kwargs: (MAEValidator(), {"src_val": "target_test"}),

    "src_train_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "src_train"}),
    "src_val_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "src_val"}),
    "target_train_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "target_train"}),
    "target_val_bnm_score": lambda kwargs: (BNMValidator(), {"target_train": "target_val"}),

    "src_train_target_train_bnm_score": lambda kwargs: (
    BNMWithSourceValidator(), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_bnm_score": lambda kwargs: (
    BNMWithSourceValidator(), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_bnm_score": lambda kwargs: (
    BNMWithSourceValidator(), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_bnm_score": lambda kwargs: (
    BNMWithSourceValidator(), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "src_train"}),
    "src_val_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "src_val"}),
    "target_train_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "target_train"}),
    "target_val_entropy_score": lambda kwargs: (EntropyValidator(), {"target_train": "target_val"}),

    "src_train_target_train_entropy_score": lambda kwargs: (EntropyCombinedValidator(), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_entropy_score": lambda kwargs: (EntropyCombinedValidator(), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_entropy_score": lambda kwargs: (EntropyCombinedValidator(), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_entropy_score": lambda kwargs: (EntropyCombinedValidator(), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_im_score": lambda kwargs: (IMValidator(), {"target_train": "src_train"}),
    "src_val_im_score": lambda kwargs: (IMValidator(), {"target_train": "src_val"}),
    "target_train_im_score": lambda kwargs: (IMValidator(), {"target_train": "target_train"}),
    "target_val_im_score": lambda kwargs: (IMValidator(), {"target_train": "target_val"}),

    "src_train_target_train_im_score": lambda kwargs: (IMCombinedValidator(), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_im_score": lambda kwargs: (IMCombinedValidator(), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_im_score": lambda kwargs: (IMCombinedValidator(), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_im_score": lambda kwargs: (IMCombinedValidator(), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(), {"target_train": "src_train"}),
    "src_val_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(), {"target_train": "src_val"}),
    "target_train_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(), {"target_train": "target_train"}),
    "target_val_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(), {"target_train": "target_val"}),

    "src_train_target_train_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(with_src=True), {"src_train": "src_train", "target_train": "target_train"}),
    "src_val_target_train_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(with_src=True), {"src_train": "src_val", "target_train": "target_train"}),
    "src_train_target_val_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(with_src=True), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_val_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(with_src=True), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits"), {"target_train": "src_train"}),
    "src_val_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits"), {"target_train": "src_val"}),
    "target_train_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits"), {"target_train": "target_train"}),
    "target_val_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits"), {"target_train": "target_val"}),

    "src_train_target_train_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits", with_src=True), {"src_train": "src_train", "target_train": "target_train"}),
    "src_val_target_train_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits", with_src=True), {"src_train": "src_val", "target_train": "target_train"}),
    "src_train_target_val_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits", with_src=True), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_val_logits_class_<score_fn_name>_score": lambda kwargs: (ClassClusterValidator(layer="logits", with_src=True), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "src_train"}),
    "src_val_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "src_val"}),
    "target_train_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "target_train"}),
    "target_val_snd_score": lambda kwargs: (SNDValidator(), {"target_train": "target_val"}),

    "src_train_target_train_mmd_score": lambda kwargs: (
    MMDValidator(mmd_kwargs=dict(mmd_type="quadratic")), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_mmd_score": lambda kwargs: (
    MMDValidator(mmd_kwargs=dict(mmd_type="quadratic")), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_mmd_score": lambda kwargs: (
    MMDValidator(mmd_kwargs=dict(mmd_type="quadratic")), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_mmd_score": lambda kwargs: (
    MMDValidator(mmd_kwargs=dict(mmd_type="quadratic")), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_logits_mmd_score": lambda kwargs: (
    MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_logits_mmd_score": lambda kwargs: (
    MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_logits_mmd_score": lambda kwargs: (
    MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_logits_mmd_score": lambda kwargs: (
    MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_preds_mmd_score": lambda kwargs: (
    MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_preds_mmd_score": lambda kwargs: (
    MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_preds_mmd_score": lambda kwargs: (
    MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_preds_mmd_score": lambda kwargs: (
    MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic")),
    {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_logits_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_logits_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_logits_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_logits_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="logits", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_preds_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_preds_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_preds_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_preds_mmd_per_class_score": lambda kwargs: (
    PerClassValidator(MMDValidator(layer="preds", mmd_kwargs=dict(mmd_type="quadratic"))),
    {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_coral_score": lambda kwargs: (CORALValidator(), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_coral_score": lambda kwargs: (CORALValidator(), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_coral_score": lambda kwargs: (CORALValidator(), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_coral_score": lambda kwargs: (CORALValidator(), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_logits_coral_score": lambda kwargs: (CORALValidator(layer="logits"), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_logits_coral_score": lambda kwargs: (CORALValidator(layer="logits"), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_logits_coral_score": lambda kwargs: (CORALValidator(layer="logits"), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_logits_coral_score": lambda kwargs: (CORALValidator(layer="logits"), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_preds_coral_score": lambda kwargs: (CORALValidator(layer="preds"), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_preds_coral_score": lambda kwargs: (CORALValidator(layer="preds"), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_preds_coral_score": lambda kwargs: (CORALValidator(layer="preds"), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_preds_coral_score": lambda kwargs: (CORALValidator(layer="preds"), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator()), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator()), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator()), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator()), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_logits_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="logits")), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_logits_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="logits")), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_logits_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="logits")), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_logits_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="logits")), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_preds_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="preds")), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_preds_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="preds")), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_preds_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="preds")), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_preds_coral_per_class_score": lambda kwargs: (PerClassValidator(CORALValidator(layer="preds")), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_rank_me_score": lambda kwargs: (RankMeValidator(with_target=False), {"src_train": "src_train"}),
    "src_val_rank_me_score": lambda kwargs: (RankMeValidator(with_target=False), {"src_train": "src_val"}),
    "target_train_rank_me_score": lambda kwargs: (RankMeValidator(with_target=False), {"src_train": "target_train"}),
    "target_val_rank_me_score": lambda kwargs: (RankMeValidator(with_target=False), {"src_train": "target_val"}),

    "src_train_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=False), {"src_train": "src_train"}),
    "src_val_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=False), {"src_train": "src_val"}),
    "target_train_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=False), {"src_train": "target_train"}),
    "target_val_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=False), {"src_train": "target_val"}),

    "src_train_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=False), {"src_train": "src_train"}),
    "src_val_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=False), {"src_train": "src_val"}),
    "target_train_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=False), {"src_train": "target_train"}),
    "target_val_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=False), {"src_train": "target_val"}),

    "src_train_target_train_rank_me_score": lambda kwargs: (RankMeValidator(with_target=True), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_rank_me_score": lambda kwargs: (RankMeValidator(with_target=True), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_rank_me_score": lambda kwargs: (RankMeValidator(with_target=True), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_rank_me_score": lambda kwargs: (RankMeValidator(with_target=True), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=True), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=True), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=True), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_logits_rank_me_score": lambda kwargs: (RankMeValidator(layer="logits", with_target=True), {"src_train": "src_val", "target_train": "target_val"}),

    "src_train_target_train_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=True), {"src_train": "src_train", "target_train": "target_train"}),
    "src_train_target_val_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=True), {"src_train": "src_train", "target_train": "target_val"}),
    "src_val_target_train_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=True), {"src_train": "src_val", "target_train": "target_train"}),
    "src_val_target_val_preds_rank_me_score": lambda kwargs: (RankMeValidator(layer="preds", with_target=True), {"src_train": "src_val", "target_train": "target_val"}),

    "target_train_henry_score": lambda kwargs: (
    HenryValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val", "target_train": "target_train"}),
    "target_val_henry_score": lambda kwargs: (
    HenryValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val", "target_train": "target_val"}),

    "target_train_henry_v3_score": lambda kwargs: (
    HenryV3RegressionValidator(), {"src_val": "src_val", "target_train": "target_train"}),
    "target_val_henry_v3_score": lambda kwargs: (
    HenryV3RegressionValidator(), {"src_val": "src_val", "target_train": "target_val"}),

    "target_train_henry_updated_score": lambda kwargs: (
    HenryUpdatedValidator(num_classes=kwargs["num_classes"], src_val_at_init=kwargs["src_val_at_init"]),
    {"src_val": "src_val", "target_train": "target_train"}),
    "target_val_henry_updated_score": lambda kwargs: (
    HenryUpdatedValidator(num_classes=kwargs["num_classes"], src_val_at_init=kwargs["src_val_at_init"]),
    {"src_val": "src_val", "target_train": "target_val"}),

    "src_train_gouk_score": lambda kwargs: (GoukValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_train"}),
    "src_val_gouk_score": lambda kwargs: (GoukValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val"}),

    "src_train_gouk_v3_score": lambda kwargs: (
    GoukV3Validator(num_classes=kwargs["num_classes"]), {"src_val": "src_train"}),
    "src_val_gouk_v3_score": lambda kwargs: (
    GoukV3Validator(num_classes=kwargs["num_classes"]), {"src_val": "src_val"}),

    "src_train_gouk_updated_score": lambda kwargs: (
    GoukUpdatedValidator(num_classes=kwargs["num_classes"], src_val_at_init=kwargs["src_val_at_init"]),
    {"src_val": "src_train"}),
    "src_val_gouk_updated_score": lambda kwargs: (
    GoukUpdatedValidator(num_classes=kwargs["num_classes"], src_val_at_init=kwargs["src_val_at_init"]),
    {"src_val": "src_val"}),

    "src_train_improved_gouk_score": lambda kwargs: (
    ImprovedGoukValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_train"}),
    "src_val_improved_gouk_score": lambda kwargs: (
    ImprovedGoukValidator(num_classes=kwargs["num_classes"]), {"src_val": "src_val"}),
}

cluster_score_fn_names = [
    "ari", "ami", "v_measure", "fmi",
    "silhouette", "chi", "dbi"
]


class Validator():
    def __init__(self, validators, main_validator, patience):
        self.validators = validators
        self.main_validator = main_validator
        self.patience = patience
        self.scores = []

    def discretize(self, data, num_classes=8, num_pred_dims=4, min_input_value=0, max_input_value=1, eps=1e-5):
        dc = int(num_classes ** (1 / num_pred_dims))
        boundaries = torch.linspace(min_input_value, max_input_value, dc + 1).to(data["target_test"]["logits"].device)
        for split in data:
            logits = torch.clamp(data[split]["logits"], min=min_input_value, max=max_input_value - eps)
            labels = torch.clamp(data[split]["labels"], min=min_input_value, max=max_input_value - eps)
            discretized_logits = torch.bucketize(logits, boundaries, right=True) - 1
            discretized_labels = torch.bucketize(labels, boundaries, right=True) - 1

            r = torch.tensor([dc ** (num_pred_dims - i - 1) for i in range(num_pred_dims)]).to(
                data["target_test"]["logits"].device)
            data[split]["class_preds"] = (discretized_logits * r).sum(dim=1)
            data[split]["class_labels"] = (discretized_labels * r).sum(dim=1)

            data[split]["class_preds"] = F.one_hot(
                data[split]["class_preds"].type(torch.LongTensor),
                num_classes=num_classes
            ).to(torch.float32)
        return data

    def compute(self, data, epoch):
        d = {"epoch": epoch + 1}
        for name, (validator, input_keys) in self.validators.items():
            new_data = deepcopy(data)
            # for all validators other than MSE/MAE, use the discretized labels
            if not any([isinstance(validator, c) for c in [MSEValidator, MAEValidator, HenryV3RegressionValidator]]):
                for split in data:
                    new_data[split]["labels"] = data[split]["class_labels"]
                    new_data[split]["preds"] = data[split]["class_preds"]
            try:
                val = validator(**{key: new_data[value] for key, value in input_keys.items()})
            except:
                val = {n: 0 for n in cluster_score_fn_names} if "<score_fn_name>" in name else 0
            if "<score_fn_name>" in name:
                for c_name in cluster_score_fn_names:
                    d[name.replace("<score_fn_name>", c_name)] = val[c_name]
            else:
                d[name] = val
        self.scores.append(d)

    def compute_validators(self, data, validators):
        d = {}
        for name, (validator, input_keys) in validators.items():
            new_data = deepcopy(data)
            # for all validators other than MSE/MAE, use the discretized labels
            if not any([isinstance(validator, c) for c in [MSEValidator, MAEValidator, HenryV3RegressionValidator]]):
                for split in data:
                    new_data[split]["labels"] = data[split]["class_labels"]
                    new_data[split]["preds"] = data[split]["class_preds"]
            try:
                val = validator(**{key: new_data[value] for key, value in input_keys.items()})
            except:
                val = {n: 0 for n in cluster_score_fn_names} if "<score_fn_name>" in name else 0
            if "<score_fn_name>" in name:
                for c_name in cluster_score_fn_names:
                    d[name.replace("<score_fn_name>", c_name)] = val[c_name]
            else:
                d[name] = val
        return d

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

    def print(self, scores=None, recent=True):
        s = scores if scores is not None else self.scores
        history = [s[-1]] if recent else s
        for scores in history:
            for k, v in scores.items():
                print(f"\t{k}:\t{v:.2f}")
        print("\n")

    def save(self, results_root, dataset, source, target, algorithm, filename, indent=4):
        with open(os.path.join(results_root, dataset, source, target, algorithm, filename), "w") as f:
            json.dump(self.scores[-1], f, indent=indent)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.algorithm == "source-only":
        args.validators = [v for v in args.validators if "mse_score" in v or "mae_score" in v]

    #src_val_at_init = torch.load(
    #    os.path.join(args.results_root, args.dataset, args.source, args.target, "source-only", "best_features.pt"))[
    #    "src_val"]

    num_classes, num_pred_dims, max_input_value = DATASETS[args.dataset][1:]
    chosen_validators = {key: VALIDATORS[key]({
        "num_classes": num_classes,
        "src_val_at_init": None,
        "dev_temp": os.path.join(".dev_temp", args.results_root, args.dataset, args.source, args.target,
                                 args.algorithm)}) for key in args.validators}
    validator = Validator(chosen_validators, "target_test_mse_score", 200)

    score_files = glob.glob(
        os.path.join(args.results_root, args.dataset, args.source, args.target, args.algorithm, "features*.pt"))
    print(
        f"Computing validators for {len(score_files)} checkpoints in {args.dataset}-{args.source[0].upper() + args.target[0].upper()} {args.algorithm} ")
    for f in score_files:
        for fl in ["fl3", "fl6"]:
            run_name = "{" + f.split("{")[1].split("}")[0] + "}"
            epoch = int(f.split("_")[-1].split(".")[0])

            scores_path = os.path.join(args.results_root, args.dataset, args.source, args.target, args.algorithm,
                                       f"post_scores_{run_name}_{epoch}_{fl}.json")
            scores_keys = []
            if os.path.exists(scores_path):
                scores = json.load(open(scores_path, "r"))
                scores_keys = [key for key in scores if key not in ["epoch"]]
                not_computed = [v for v in args.validators if v not in scores_keys]
                # remove clustering validators if computed
                computed_clustering_scores = [s for s in cluster_score_fn_names if f"src_val_target_val_logits_class_{s}_score" in scores_keys]
                if len(computed_clustering_scores) == len(cluster_score_fn_names):
                    for v in not_computed:
                        not_computed = [v for v in not_computed if "<score_fn_name>" not in v]

                if len(not_computed) == 0:
                    print(
                        f"{args.dataset}-{args.source[0].upper() + args.target[0].upper()} {args.algorithm} {run_name}_{epoch}_{fl} already computed all validators, skipping.")
                    continue
            else:
                not_computed = args.validators

            if not os.path.exists(scores_path):
                print(f"Validating all for {args.dataset}-{args.source[0].upper() + args.target[0].upper()} {args.algorithm} {run_name}_{epoch}_{fl}.")
                eval_data = torch.load(f)
                for key in eval_data:
                    eval_data[key]["features"] = eval_data[key][fl]
                eval_data = validator.discretize(eval_data, num_classes=num_classes, num_pred_dims=num_pred_dims,
                                                 max_input_value=max_input_value)
                eval_data = {key: {key2: val2.to("cuda:0") for key2, val2 in val.items()} for key, val in
                             eval_data.items()}
                validator.compute(eval_data, epoch - 1)
                validator.print()
                validator.save(args.results_root, args.dataset, args.source, args.target, args.algorithm,
                               f"post_scores_{run_name}_{epoch}_{fl}.json")
            else:
                print(
                    f"Validating {not_computed} for {args.dataset}-{args.source[0].upper() + args.target[0].upper()} {args.algorithm} {run_name}_{epoch}_{fl}.")
                eval_data = torch.load(f)
                for key in eval_data:
                    eval_data[key]["features"] = eval_data[key][fl]
                eval_data = validator.discretize(eval_data, num_classes=num_classes, num_pred_dims=num_pred_dims,
                                                 max_input_value=max_input_value)
                eval_data = {key: {key2: val2.to("cuda:0") for key2, val2 in val.items()} for key, val in
                             eval_data.items()}
                uncomputed_validators = {key: VALIDATORS[key]({
                    "num_classes": num_classes,
                    "src_val_at_init": None,
                    "dev_temp": os.path.join(".dev_temp", args.results_root, args.dataset, args.source, args.target,
                                             args.algorithm)}) for key in not_computed}
                print(uncomputed_validators)
                validator_scores = validator.compute_validators(eval_data, uncomputed_validators)
                print(validator_scores)
                validator.scores.append(json.load(
                    open(os.path.join(
                        args.results_root, args.dataset, args.source, args.target, args.algorithm,
                        f"post_scores_{run_name}_{epoch}_{fl}.json"
                    ), "r")
                )
                )
                validator.scores[-1].update(validator_scores)
                validator.print()
                validator.save(args.results_root, args.dataset, args.source, args.target, args.algorithm,
                               f"post_scores_{run_name}_{epoch}_{fl}.json")
