import os
import re
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models import PatchTSTClassifier
from tools import *  # set_seed, compute_f1_score, write_result 等
from PatchTST_test import test_model_with_path_tracking


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                save_path, fig_path, num_epochs=150, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0
    patience_counter = 0

    train_losses, train_f1_scores, val_f1_scores = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)      # [B, num_classes]
            preds = (probs > 0.5).int()         # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, "
              f"Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Model Saved (Best F1-score)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping Triggered")
                break

    # 繪圖
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='s')
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss & F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


# 依錄影資料夾名稱取得 sample indices
def get_indices_by_recording(dataset, recording_list):
    indices = []
    for idx in range(len(dataset)):
        sample_path = dataset.get_sample_path(idx)
        for rec in recording_list:
            if rec in sample_path:
                indices.append(idx)
                break
    return indices


# 解析 per_class_f1_scores.txt
def parse_per_class_file(path):
    vals = {}
    if not os.path.exists(path):
        print(f"[WARN] per-class F1 file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Class\s+(\d+):\s*([0-9]*\.?[0-9]+)", line)
            if m:
                c = int(m.group(1))
                v = float(m.group(2))
                vals[c] = v
    if not all(k in vals for k in [0, 1, 2, 3]):
        print(f"[WARN] per-class F1 file missing some classes: {path} -> {vals}")
        return None
    return vals


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, required=True)
    parser.add_argument('--master_seed', type=int, default=539,
                        help='seed used only to partition 10 disjoint test folds')
    args = parser.parse_args()

    from dataset.PatchTST import *
    # 依運動類型載入資料集 2D_traindata_Final 3D_Real_Final
    if args.sport == 'deadlift':
        dataset_path = os.path.join(os.getcwd(), 'data', '2D_traindata_Final')
        full_dataset = Dataset_TST_Deadlift(dataset_path)
        save_dir = './models/TST_Deadlift/2D_Real_new_3'
        num_classes = 4
        input_len = 110
    elif args.sport == 'benchpress':
        dataset_path = os.path.join(os.getcwd(), 'data', 'BP_data_oringinal', 'bench_press_multilabel_cut4.csv')
        full_dataset = Dataset_TST_Benchpress(dataset_path)
        save_dir = './models/benchpress/TST_Benchpress/6'
        num_classes = 4
        input_len = 100
    else:
        raise ValueError("Unsupported sport")

    os.makedirs(save_dir, exist_ok=True)

    input_dim = full_dataset.dim
    print('Input dimension:', input_dim)

    # ========= 用 master_seed 切 10 個互斥 test folds（每 fold 16 位錄影資料夾） =========
    all_recordings = list(full_dataset.data.keys())
    required_total = 16 * 10
    if len(all_recordings) < required_total:
        raise ValueError(f"錄影資料夾（受試者）數不足：需要至少 {required_total}，目前僅有 {len(all_recordings)}")

    rng = random.Random(args.master_seed)
    rng.shuffle(all_recordings)
    folds = [sorted(all_recordings[i*16:(i+1)*16]) for i in range(10)]
    flat = [r for fold in folds for r in fold]
    assert len(flat) == required_total, "folds 展開後總數不為 160，請檢查切分邏輯。"
    assert len(set(flat)) == required_total, "不同 folds 之間有重複錄影資料夾，請檢查。"

    print("\n📌 Disjoint test folds (preview):")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: {len(fold)} recs, e.g., {fold[:3]} ...")

    # ========= 跑 10 次實驗 =========
    # seeds = [44, 2025, 11, 90, 102, 1001, 2015, 835, 75, 124]
    seeds = [835,835,835,835,835,835,835,835,835,835,]
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []

    # A) per-class F1 收集容器（四類，各 10 筆）
    per_class_runs = {0: [], 1: [], 2: [], 3: []}

    for run_idx, se in enumerate(seeds):
        set_seed(se)

        # 當次 test fold（互斥 16 位）
        test_recordings = folds[run_idx]
        trainval_recordings = [rec for rec in all_recordings if rec not in test_recordings]

        print(f"\n📦 Run {run_idx+1} | Seed {se}")
        print(f"  ➤ Test recordings ({len(test_recordings)}): {test_recordings[:6]} ...")

        # 由錄影清單轉成 sample indices
        test_indices = get_indices_by_recording(full_dataset, test_recordings)
        trainval_indices = get_indices_by_recording(full_dataset, trainval_recordings)

        # 在 trainval（以 sample 為單位）切出 val（固定 15%）
        train_indices, val_indices = train_test_split(
            trainval_indices, test_size=0.15, random_state=se
        )

        # Dataloaders
        train_dataset = TransformSubset(full_dataset, train_indices, transform=True)
        val_dataset   = TransformSubset(full_dataset, val_indices,   transform=False)
        test_dataset  = TransformSubset(full_dataset, test_indices,  transform=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

        # Model / Optim
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

        # 路徑包含 fold id，便於辨識
        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}_fold{run_idx}.pth")
        txt_dir   = os.path.join(save_dir, f"PatchTST_model_seed{se}_fold{run_idx}_results")
        fig_path  = os.path.join(txt_dir,   f"train_results_seed{se}_fold{run_idx}.png")
        os.makedirs(txt_dir, exist_ok=True)

        # 訓練
        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        # 測試（會在 txt_dir 內輸出 per_class_f1_scores.txt）
        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, full_dataset, num_classes
        )
        print(f"Seed {se} (Fold {run_idx}) Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, "
              f"cost {avg_time_per_sample:.4f} sec")

        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

        # B) 讀本回合 per-class F1 檔案並收集
        per_class_file = os.path.join(txt_dir, "per_class_f1_scores.txt")
        parsed = parse_per_class_file(per_class_file)
        if parsed is not None:
            for c in [0, 1, 2, 3]:
                per_class_runs[c].append(parsed[c])

    # C) 迴圈後：計算 per-class 平均與標準差並寫出到 save_dir
    summary_path = os.path.join(save_dir, "per_class_f1_mean_std.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Per-class F1 over 10 runs (mean ± std)\n")
        for c in [0, 1, 2, 3]:
            arr = np.array(per_class_runs[c], dtype=float)
            if arr.size == 0:
                f.write(f"Class {c}: N/A (no data)\n")
            else:
                mu = arr.mean()
                sd = arr.std(ddof=1) if arr.size > 1 else 0.0
                f.write(f"Class {c}: {mu:.4f} ± {sd:.4f}  (n={arr.size})\n")
    print(f"[OK] Wrote per-class F1 mean/std to: {summary_path}")

    # 總結
    write_result(model, seeds, all_f1_scores, accuracies, cost_times,
                 save_dir, best_f1, best_seed, best_model_path)
