import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import h5py
import wandb
import random
from math import ceil
import model
from collections import Counter


# Config 

wandb.init(project="pcd_cls", entity="xlab", config={
    "learning_rate": 0.005,
    "architecture": "PointNet++",
    "dataset": "cubes_pc_pla_reduced_sectioned_shaved",
    "epochs": 100,
    "batch_size": 160,
    "target_points": 100000,
})

# Grouping rule (class labels come from this)
GROUP_CHAR_INDEX = 0                  # which character to use from identifier (0 = first)
GROUP_VALUES     = ['1', '2', '3', '4', '5']    # valid classes
INCLUDE_OTHER    = False             
VAL_RATIO        = 0.20               # per-group fraction for validation (the rest goes to train)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
HDF5_PATH = '/srv/scratch/arashid/datasets/test/cubes_pc_pla_reduced_sectioned_shaved.h5'             # add directory to the h5 file

print(f"Classification classes: {GROUP_VALUES}  |  char index: {GROUP_CHAR_INDEX}")
print(f"Split: train={(1.0-VAL_RATIO):.2f}  val={VAL_RATIO:.2f}")


# Optional: GPU memory print helper

def _bytes_to_gb(x): return x / (1024**3)
def print_gpu_memory(title=None, device_ids=None):
    if title: print(title)
    if not torch.cuda.is_available():
        print("  [CPU only]")
        return
    if device_ids is None:
        try:
            device_ids = net.device_ids  
        except Exception:
            device_ids = list(range(torch.cuda.device_count()))
    for d in device_ids:
        alloc = torch.cuda.memory_allocated(d)
        reserv = torch.cuda.memory_reserved(d)
        max_alloc = torch.cuda.max_memory_allocated(d)
        print(f"  GPU {d}: allocated={_bytes_to_gb(alloc):.2f} GB, "
              f"reserved={_bytes_to_gb(reserv):.2f} GB, "
              f"max_alloc={_bytes_to_gb(max_alloc):.2f} GB (this process)")


# Data utils

def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    if pc.size == 0:
        return pc
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_norm = np.max(np.linalg.norm(pc, axis=1))
    if max_norm > 0:
        pc = pc / max_norm
    return pc


def resample_to_k(pc: np.ndarray, k: int) -> np.ndarray:
    n = pc.shape[0]
    if n == 0:
        return pc
    if n == k:
        return pc

    replace = (n < k)
    idx = np.random.choice(n, size=k, replace=replace)
    return pc[idx]



# Load HDF5 in one pass (resample every chunk to fixed target_points)

target_points = int(wandb.config.target_points)
print(f"Resampling every chunk to target_points = {target_points}")

pointclouds, part_names = [], []

with h5py.File(HDF5_PATH, "r") as h5f:
    for chunk_name in h5f.keys():
        part_base = chunk_name.split('_', 1)[0]

        pc = np.array(h5f[chunk_name])
        if pc is None or pc.size == 0:
            continue

        pc = normalize_pointcloud(pc)
        pc = resample_to_k(pc, target_points)

        pointclouds.append(pc.T.astype(np.float32))
        part_names.append(chunk_name)

if len(pointclouds) == 0:
    raise ValueError("No valid point clouds found in HDF5 after loading/resampling.")

pointclouds = np.stack(pointclouds, axis=0).astype(np.float32)
print(f"Loaded {len(pointclouds)} point-cloud chunks from {HDF5_PATH} (all resampled to {target_points})")



# Build identifier -> indices

identifier_dict = defaultdict(list)
for i, name in enumerate(part_names):
    ident = name.split('_')[0]
    identifier_dict[ident].append(i)


# Group identifiers by the target character

group_to_identifiers = {v: [] for v in GROUP_VALUES}
others_identifiers = []
for ident in identifier_dict:
    ch = ident[GROUP_CHAR_INDEX] if len(ident) > GROUP_CHAR_INDEX else None
    if ch in group_to_identifiers:
        group_to_identifiers[ch].append(ident)
    else:
        others_identifiers.append(ident)

if INCLUDE_OTHER and len(others_identifiers) > 0 and 'OTHER' not in group_to_identifiers:
    group_to_identifiers['OTHER'] = others_identifiers


# Split helper (per-group)

def split_identifiers_ratio(id_list, ratio):
    id_list = id_list[:]
    random.shuffle(id_list)
    n = len(id_list)
    n_val = max(1, int(round(n * ratio))) if ratio > 0 else 0
    n_val = min(n_val, n)
    val_ids = id_list[:n_val]
    train_ids = id_list[n_val:]
    return train_ids, val_ids


# Per-group split to train/val

train_indices, val_indices = [], []
for g, id_list in group_to_identifiers.items():
    if len(id_list) == 0:  
        continue
    train_ids, val_ids = split_identifiers_ratio(id_list, VAL_RATIO)
    for vid in val_ids:
        val_indices.extend(identifier_dict[vid])
    for trid in train_ids:
        train_indices.extend(identifier_dict[trid])

# Summary (identifiers)
print("\n=== Identifier split summary (train/val) ===")
for g in group_to_identifiers:
    ids = group_to_identifiers[g]
    tr = [i for i in ids if i in identifier_dict and any(idx in train_indices for idx in identifier_dict[i])]
    va = [i for i in ids if i in identifier_dict and any(idx in val_indices   for idx in identifier_dict[i])]
    print(f"Group '{g}': total_id={len(ids)} | train_id={len(tr)} | val_id={len(va)}")
print("============================================\n")


# Print summary: samples and chunk counts per split

from collections import Counter

chunk_counts = Counter(name.split('_',1)[0] for name in part_names)

train_idents_final = set(part_names[i].split('_',1)[0] for i in train_indices)
val_idents_final   = set(part_names[i].split('_',1)[0] for i in val_indices)

print("\n=== Split Overview ===")
print(f"Train samples: {len(train_idents_final)}   | Val samples: {len(val_idents_final)}")
print(f"Train chunks : {len(train_indices)}        | Val chunks : {len(val_indices)}")

print("\n--- TRAIN SAMPLES ---")
for ident in sorted(train_idents_final):
    print(f"{ident}: {chunk_counts[ident]} chunks")

print("\n--- VALIDATION SAMPLES ---")
for ident in sorted(val_idents_final):
    print(f"{ident}: {chunk_counts[ident]} chunks")
print("============================\n")




# Build class labels from identifiers

classes = list(group_to_identifiers.keys())
class_to_idx = {c: i for i, c in enumerate(classes)}

def index_to_class_idx(sample_index: int) -> int:
    ident = part_names[sample_index].split('_')[0]
    ch = ident[GROUP_CHAR_INDEX] if len(ident) > GROUP_CHAR_INDEX else None
    label = ch if ch in class_to_idx else ('OTHER' if INCLUDE_OTHER else None)
    if label is None:  
        return -1
    return class_to_idx[label]

# Filter out-of-class samples
train_indices = [i for i in train_indices if index_to_class_idx(i) != -1]
val_indices   = [i for i in val_indices   if index_to_class_idx(i) != -1]

# Final tensors
train_pcd = pointclouds[train_indices]
val_pcd   = pointclouds[val_indices]
train_y   = np.array([index_to_class_idx(i) for i in train_indices], dtype=np.int64)
val_y     = np.array([index_to_class_idx(i) for i in val_indices],   dtype=np.int64)

print(f"Final sizes -> train: {len(train_pcd)} | val: {len(val_pcd)}")
print(f"Classes: {classes}  (mapping: {class_to_idx})")


class PointCloudDatasetCls(Dataset):
    def __init__(self, pcd, labels_int, orig_indices):
        self.pcd = torch.tensor(pcd, dtype=torch.float32)     # (B, 3, N)
        self.labels = torch.tensor(labels_int, dtype=torch.long)
        self.orig_indices = np.array(orig_indices, dtype=np.int64)

    def __len__(self): 
        return len(self.pcd)

    def __getitem__(self, idx): 
        return self.pcd[idx], self.labels[idx], int(self.orig_indices[idx])


batch_size = wandb.config.batch_size

train_loader = DataLoader(
    PointCloudDatasetCls(train_pcd, train_y, train_indices),
    batch_size=batch_size, shuffle=True, drop_last=True,
)

val_loader = DataLoader(
    PointCloudDatasetCls(val_pcd, val_y, val_indices),
    batch_size=batch_size, shuffle=False
)


point_size = train_pcd.shape[2]


# Device, model, optimizer, AMP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(classes)
net = model.PointNet2Cls_59(point_size, latent_size=512, num_classes=num_classes)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    net = nn.DataParallel(net)
net = net.to(device)

print_gpu_memory("GPU memory before training begins")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=wandb.config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda')

num_epochs = wandb.config.epochs


def evaluate(loader, split_name: str):
    net.eval()
    total_correct, total_count = 0, 0
    per_group = {c: {"correct": 0, "total": 0} for c in classes}
    all_true, all_pred, all_idx = [], [], []
    with torch.no_grad():
        for points, labels, idxs in loader:
            points = points.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda'):
                logits = net(points)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_count   += labels.numel()

            t_cpu = labels.cpu().tolist()
            p_cpu = preds.cpu().tolist()
            i_cpu = list(map(int, idxs))

            all_true.extend(t_cpu)
            all_pred.extend(p_cpu)
            all_idx.extend(i_cpu)

            for p, t in zip(p_cpu, t_cpu):
                cname = classes[t]
                per_group[cname]["total"] += 1
                if p == t:
                    per_group[cname]["correct"] += 1

    acc = (total_correct / total_count) if total_count > 0 else 0.0
    for cname, d in per_group.items():
        d["acc"] = (d["correct"] / d["total"]) if d["total"] > 0 else 0.0

    print(f"[{split_name}] correct={total_correct}/{total_count}  ({acc*100:.2f}%)")
    per_line = "  ".join([f"{k}: {v['correct']}/{v['total']} ({v['acc']*100:.2f}%)"
                           for k, v in per_group.items()])
    print(f"[{split_name}] per-group → {per_line}")

    return {
        "total_correct": total_correct,
        "total_count": total_count,
        "acc": acc,
        "per_group": per_group,
        "y_true": all_true,
        "y_pred": all_pred,
        "sample_indices": all_idx,
    }



# Train/Eval loop

best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nStarting epoch {epoch+1}/{num_epochs} ...")
    print_gpu_memory(f"Memory snapshot at start of epoch {epoch+1}")
    net.train()
    running_loss = 0.0
    samples_in_epoch = 0

    for batch_idx, (points, labels, _) in enumerate(train_loader):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = net(points)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        samples_in_epoch += labels.size(0)
        if batch_idx % 10 == 0:
            total_batches = len(train_loader)
            print(f"[Epoch {epoch+1}] batch {batch_idx+1}/{total_batches}  loss={loss.item():.6f}")
            wandb.log({"batch_loss": loss.item(), "epoch": epoch})

    avg_train_loss = running_loss / max(1, samples_in_epoch)
    wandb.log({"average_training_loss": avg_train_loss, "epoch": epoch})

    train_metrics = evaluate(train_loader, "Train")
    val_metrics   = evaluate(val_loader,   "Validation")

    if val_metrics["acc"] > best_val_acc:
        best_val_acc = val_metrics["acc"]

        import pandas as pd
        ctab = pd.crosstab(
            pd.Series(val_metrics["y_true"], name="true"),
            pd.Series(val_metrics["y_pred"], name="pred")
        )

        table_rows = []
        for t_idx, t_name in enumerate(classes):
            for p_idx, p_name in enumerate(classes):
                cnt = int(ctab.reindex(index=range(len(classes)), columns=range(len(classes))).fillna(0).iloc[t_idx, p_idx]) \
                    if not ctab.empty else 0
                table_rows.append([t_name, p_name, cnt])
        wb_table = wandb.Table(columns=["true", "pred", "count"], data=table_rows)
        wandb.log({"best_pred_vs_true_counts": wb_table, "epoch": epoch})

        y_true_int = [int(x) for x in val_metrics["y_true"]]
        y_pred_int = [int(x) for x in val_metrics["y_pred"]]

        wandb.log({
            "best_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true_int,
                preds=y_pred_int,
                class_names=classes  # ['1','2','3']
            ),
            "epoch": epoch
        })

        run_name = wandb.run.name if wandb.run and wandb.run.name else f"run_{wandb.run.id}"
        out_dir = os.path.join(os.path.dirname(HDF5_PATH), "pcd_output")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{run_name}.csv")

        rows = []
        for idx, t, p in zip(val_metrics["sample_indices"], val_metrics["y_true"], val_metrics["y_pred"]):
            rows.append({
                "chunk_name": part_names[idx],
                "true_class": classes[t],
                "pred_class": classes[p],
                "correct": int(p == t)
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved best-val CSV to: {out_csv}")


    wandb.log({
        "train_acc": train_metrics["acc"],
        "best_val_acc": best_val_acc,
        "train_correct": train_metrics["total_correct"],
        "train_total": train_metrics["total_count"],
        "val_acc": val_metrics["acc"],
        "val_correct": val_metrics["total_correct"],
        "val_total": val_metrics["total_count"],
        "epoch": epoch
    })

    for cname in classes:
        wandb.log({
            f"train_acc_{cname}": train_metrics["per_group"][cname]["acc"],
            f"val_acc_{cname}":   val_metrics["per_group"][cname]["acc"],
            "epoch": epoch
        })

    scheduler.step(val_metrics["acc"])

    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr, "epoch": epoch})
    print(f"Best validation accuracy so far: {best_val_acc*100:.2f}%")
    print(f"Finished epoch {epoch+1}/{num_epochs}")
    print_gpu_memory(f"Memory snapshot after epoch {epoch+1}")

wandb.finish()
