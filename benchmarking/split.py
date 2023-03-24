import os, sys
from glob import glob
import torch

root, domain, train_ratio, val_ratio = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

classes = sorted([name for name in os.listdir(os.path.join(root, domain)) if os.path.isdir(os.path.join(root, domain, name))])
filenames = glob(f"{os.path.join(root, domain)}/**/*.jpg", recursive=True)
n = len(filenames)
perm = torch.randperm(n)

train_idx = perm[:int(train_ratio * n)]
val_idx = perm[int(train_ratio * n):int(train_ratio * n) + int(val_ratio * n)]
trainval_idx = torch.cat([train_idx, val_idx])
test_idx = perm[int(train_ratio * n) + int(val_ratio * n):]

print(n, len(train_idx), len(val_idx), len(trainval_idx), len(test_idx), len(perm))

with open(f"{os.path.join(root, domain)}/train.txt", "w") as f:
    for i in train_idx:
        f.write(f"{'/'.join(filenames[i].split('/')[-3:])} {classes.index(filenames[i].split('/')[-2])}\n")
with open(f"{os.path.join(root, domain)}/val.txt", "w") as f:
    for i in val_idx:
        f.write(f"{'/'.join(filenames[i].split('/')[-3:])} {classes.index(filenames[i].split('/')[-2])}\n")
with open(f"{os.path.join(root, domain)}/trainval.txt", "w") as f:
    for i in trainval_idx:
        f.write(f"{'/'.join(filenames[i].split('/')[-3:])} {classes.index(filenames[i].split('/')[-2])}\n")
with open(f"{os.path.join(root, domain)}/test.txt", "w") as f:
    for i in test_idx:
        f.write(f"{'/'.join(filenames[i].split('/')[-3:])} {classes.index(filenames[i].split('/')[-2])}\n")
with open(f"{os.path.join(root, domain)}/all.txt", "w") as f:
    for i in perm:
        f.write(f"{'/'.join(filenames[i].split('/')[-3:])} {classes.index(filenames[i].split('/')[-2])}\n")
