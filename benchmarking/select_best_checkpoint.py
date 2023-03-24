import os, sys, glob, json, shutil
import numpy as np
import pandas as pd
import torch

results_root, dataset, source, target, algorithm, validator = sys.argv[1:7]

score_files = glob.glob(os.path.join(results_root, dataset, source, target, algorithm, "scores*.json"))
d = {filename.split("/")[-1][len("scores_"):-len(".json")]: json.load(open(filename, "r")) for filename in score_files if "hpo_scores.json" not in filename}
best = {"key": None, "score": {validator: -np.inf, "epoch": np.inf}}
for key, score_dict in d.items():
    if score_dict[validator] > best["score"][validator]:
        best["score"] = score_dict
        best["key"] = key
    elif score_dict[validator] == best["score"][validator] and score_dict["epoch"] < best["score"]["epoch"]:
        best["score"] = score_dict
        best["key"] = key
print(best)
shutil.copy(os.path.join(results_root, dataset, source, target, algorithm, f"checkpoint_{best['key']}.pt"), os.path.join(results_root, dataset, source, target, algorithm, f"best.pt"))
shutil.copy(os.path.join(results_root, dataset, source, target, algorithm, f"features_{best['key']}.pt"), os.path.join(results_root, dataset, source, target, algorithm, f"best_features.pt"))
