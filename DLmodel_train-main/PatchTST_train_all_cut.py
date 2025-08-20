import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
from PatchTST_test import test_model_with_path_tracking
from sklearn.model_selection import train_test_split

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0  # 用來儲存最佳 F1-score
    patience_counter = 0

    # **存放訓練過程的數據**
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

            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # **紀錄數據**
        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # 根據 F1-score 儲存最佳模型
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

    # **繪製 Loss 和 F1-score**
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
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片

# 🔧 補充 function：根據錄影資料夾名稱取得 index
def get_indices_by_recording(dataset, recording_list):
    indices = []
    for idx in range(len(dataset)):
        sample_path = dataset.get_sample_path(idx)
        for rec in recording_list:
            if rec in sample_path:
                indices.append(idx)
                break
    return indices


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    args = parser.parse_args()
    
    from dataset.PatchTST import *

    if args.sport == 'deadlift':
        dataset_path = os.path.join(os.getcwd(), 'data', '3D_Real_Final')
        full_dataset = Dataset_TST_Deadlift(dataset_path)
        save_dir = f'./models/TST_Deadlift/19'
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

    input_dim = full_dataset.dim
    print('Input dimension:', input_dim)

    # 🔥 以錄影資料夾為單位切分
    all_recordings = list(full_dataset.data.keys())  # 資料夾名稱
    train_recs, testval_recs = train_test_split(all_recordings, test_size=0.25, random_state=42)
    val_recs, test_recs = train_test_split(testval_recs, test_size=0.4, random_state=42)  # 最後 10% 為 test

    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    seeds = [42, 2023, 7, 88, 100, 999, 1717, 3030]

    for se in seeds:
        set_seed(se)

        # ✅ 以錄影為單位取 index
        train_indices = get_indices_by_recording(full_dataset, train_recs)
        val_indices   = get_indices_by_recording(full_dataset, val_recs)
        test_indices  = get_indices_by_recording(full_dataset, test_recs)

        # ✅ 加入這段列印，幫你檢查分組情況
        print(f"\n🟦 Seed {se} - Data Split Overview:")
        print(f"  ➤ Train: {len(train_indices)} samples from {len(train_recs)} recordings")
        print(f"  ➤ Val:   {len(val_indices)} samples from {len(val_recs)} recordings")
        print(f"  ➤ Test:  {len(test_indices)} samples from {len(test_recs)} recordings")

        def print_recording_names(name, indices):
            recordings = set()
            for idx in indices:
                path = full_dataset.get_sample_path(idx)
                rec = path.split(os.sep)[-2]  # 取得倒數第二層資料夾（錄影名）
                recordings.add(rec)
            print(f"    {name} contains recordings: {sorted(list(recordings))}")

        print_recording_names("Train", train_indices)
        print_recording_names("Val", val_indices)
        print_recording_names("Test", test_indices)

        # 檢查有沒有重疊
        def check_overlap(a, b, name1, name2):
            overlap = set(a) & set(b)
            if overlap:
                print(f"  ❌ Overlap found between {name1} and {name2}: {len(overlap)} samples")
            else:
                print(f"  ✅ No overlap between {name1} and {name2}")

        check_overlap(train_indices, val_indices, "train", "val")
        check_overlap(train_indices, test_indices, "train", "test")
        check_overlap(val_indices, test_indices, "val", "test")

        train_dataset = TransformSubset(full_dataset, train_indices, transform=True)
        valid_dataset = TransformSubset(full_dataset, val_indices, transform=False)
        test_dataset  = TransformSubset(full_dataset, test_indices, transform=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        txt_dir   = os.path.join(save_dir, f"PatchTST_model_seed{se}_results")
        fig_path  = os.path.join(txt_dir, f"train_results_seed{se}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, full_dataset, num_classes
        )
        print(f"Seed {se} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample:.4f} sec")

        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_result(model, seeds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path)
