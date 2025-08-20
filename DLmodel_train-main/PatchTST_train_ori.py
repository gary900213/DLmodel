import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
from PatchTST_test_ori import test_model_with_path_tracking
import re  # <-- 解析 per-class F1 用

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0
    patience_counter = 0

    train_losses = []
    train_f1_scores = []
    val_f1_scores = []

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

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

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


def parse_per_class_f1(path):
    """讀取 per_class_f1_scores.txt -> dict{class_id: f1}"""
    vals = {}
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Class\s+(\d+):\s*([0-9]*\.?[0-9]+)", line)
            if m:
                c = int(m.group(1))
                v = float(m.group(2))
                vals[c] = v
    return vals


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    args = parser.parse_args()
    
    from dataset.PatchTST import *
    if args.sport == 'deadlift':
        dataset = os.path.join(os.getcwd(), 'data', '2D_traindata_Final')
        full_dataset = Dataset_TST_Deadlift(dataset)
        save_dir = f'./models/TST_Deadlift/2D_ori'
        num_classes = 4
        input_len = 110
    elif args.sport == 'benchpress':
        dataset = os.path.join(os.getcwd(), 'data', 'BP_data_oringinal', 'bench_press_multilabel_cut4.csv')
        full_dataset = Dataset_TST_Benchpress(dataset)
        save_dir = './models/benchpress/TST_Benchpress/6'
        num_classes = 4
        input_len = 100
    input_dim = full_dataset.dim
    print('Input dimention', input_dim)
    
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    seeds = [42, 2023, 7, 88, 100, 999, 1717, 3030]

    # ★ 這裡新增容器：存每類 across seeds 的 F1
    per_class_runs = {c: [] for c in range(num_classes)}

    for se in seeds:
        set_seed(se)

        gen = torch.Generator().manual_seed(se)
        train_indices, valid_indices, test_indices = random_split(
            range(len(full_dataset)), [train_size, valid_size, test_size],
            generator=gen
        )
        
        train_dataset = TransformSubset(full_dataset, train_indices, transform=True)
        valid_dataset = TransformSubset(full_dataset, valid_indices, transform=False)
        test_dataset  = TransformSubset(full_dataset, test_indices, transform=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        txt_dir = os.path.join(save_dir, f"PatchTST_model_seed{se}_results")
        fig_path = os.path.join(txt_dir, f"train_results_seed{se}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, full_dataset, num_classes
        )
        print(f"Seed {se} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

        # ★ 讀取這個 seed 的 per_class F1
        f1_file = os.path.join(txt_dir, "per_class_f1_scores.txt")
        parsed = parse_per_class_f1(f1_file)
        if parsed:
            for c, v in parsed.items():
                per_class_runs[c].append(v)

    # ★ 最後輸出每類 across seeds 的 mean ± std
    summary_path = os.path.join(save_dir, "per_class_f1_mean_std.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Per-class F1 over all seeds (mean ± std)\n")
        for c in range(num_classes):
            arr = np.array(per_class_runs[c], dtype=float)
            if arr.size == 0:
                f.write(f"Class {c}: N/A\n")
            else:
                mu = arr.mean()
                sd = arr.std(ddof=1) if arr.size > 1 else 0.0
                f.write(f"Class {c}: {mu:.4f} ± {sd:.4f}  (n={arr.size})\n")
    print(f"[OK] Wrote per-class F1 mean/std to: {summary_path}")

    write_result(model, seeds, all_f1_scores, accuracies, cost_times, save_dir,
                 best_f1, best_seed, best_model_path)
