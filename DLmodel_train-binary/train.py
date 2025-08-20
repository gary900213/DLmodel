import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import argparse
import random
from models import ResNet32, BiLSTMModel
from tools import set_seed, f1_score, compute_f1_score, write_results
from test import test_model_with_path_tracking
from dataset import *


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=100, patience=8):
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

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred)
        val_f1 = compute_f1_score(model, valid_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_class', type=int)
    parser.add_argument('--SHAP', type=str, default=None)
    parser.add_argument('--F_type', type=str)
    parser.add_argument('--model', type=str, default='Resnet32', choices=['ResNet32', 'BiLSTM'])
    parser.add_argument('--data', type=str)
    parser.add_argument('--sport', type=str, default='benchpress')
    parser.add_argument('--master_seed', type=int, default=830, help='seed used only to partition 10 disjoint test folds')
    args = parser.parse_args()

    GT_class = args.GT_class
    SHAP_mode = args.SHAP
    F_type = args.F_type
    model_type = args.model
    data = args.data
    sport = args.sport
    master_seed = args.master_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SHAP_mode is None:
        if F_type == '2D':
            if sport == 'deadlift':
                class_names = {
                    2: 'Barbell_moving_away_from_the_shins',
                    3: 'Hips_rising_before_the_barbell_leaves_the ground',
                    4: 'Barbell_colliding_with_the_knees',
                    5: 'Lower_back_rounding'
                }
                data_path = os.path.join(os.getcwd(), 'data', data)
                full_dataset = Dataset_dd2voz(data_path, GT_class)
                save_dir = os.path.join(os.getcwd(), 'models', 'deadlift','4', model_type, data, str(GT_class))
                category_ratio = full_dataset.get_ratio()
                P_ratio = category_ratio[str(GT_class)]

            if sport == 'benchpress':
                class_names = {
                    0: 'tilting_to_the_left',
                    1: 'tilting_to_the_right',
                    2: 'elbows_flaring',
                    3: 'scapular_protraction'
                }
                data_path = os.path.join(os.getcwd(), 'data', data, 'bench_press_multilabel_cut4.csv')
                full_dataset = Dataset_Benchpress(data_path, GT_class)
                save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data, class_names[GT_class])
                os.makedirs(save_dir, exist_ok=True)
                category_ratio = full_dataset.get_ratio()
                P_ratio = category_ratio[1]

        elif F_type == '3D':
            data_path = os.path.join(os.getcwd(), 'data', '3D_Real_Final')
            full_dataset = Dataset_3D(data_path, GT_class)
            save_dir = os.path.join(os.getcwd(), 'models', '3D_Final_Resnet32', str(GT_class))

        input_dim = full_dataset.dim

    else:
        from dataset import Dataset_SHAP
        datasets_path = os.path.join(os.getcwd(), 'dataset')
        full_dataset = Dataset_SHAP(datasets_path, GT_class, SHAP_mode)
        input_dim = full_dataset.dim
        save_dir = os.path.join(os.getcwd(), 'models_SHAP', f'{GT_class}', f'SHAP_{SHAP_mode}')

    print(f'Category : {category_ratio}')

    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    all_avg_times = []
    all_acc = []

    seeds = [43, 2024, 8, 89, 101, 1000, 2013, 833, 74, 123]

    # === 先建立 10 個互斥的 test folds（每 fold 16 位受試者） ===
    group_map = full_dataset.get_group_map()
    all_group_names = list(group_map.keys())

    required_total = 16 * 10
    if len(all_group_names) < required_total:
        raise ValueError(
            f"受試者（group）總數不足：需要至少 {required_total}，但目前只有 {len(all_group_names)}"
        )

    rng = random.Random(master_seed)
    rng.shuffle(all_group_names)
    folds = [set(all_group_names[i*16:(i+1)*16]) for i in range(10)]

    flat = [g for s in folds for g in s]
    assert len(flat) == required_total, "分割的 folds 總數不為 160，請檢查切分邏輯。"
    assert len(set(flat)) == required_total, "folds 之間有重複受試者，請檢查。"

    print("\n📌 Disjoint test folds (preview):")
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f)} groups, e.g., {sorted(list(f))[:3]} ...")

    # === 開始 10 次實驗 ===
    for run_idx, se in enumerate(seeds):
        set_seed(se)

        test_group_names = folds[run_idx]
        print(f"\n📦 Run {run_idx+1} | Seed {se} - Selected Test Groups:")
        for g in sorted(test_group_names):
            print(f"  - {g}")

        test_indices = []
        remaining_indices = []
        for group, indices in group_map.items():
            if group in test_group_names:
                test_indices.extend(indices)
            else:
                remaining_indices.extend(indices)

        train_size = int(0.8333 * len(remaining_indices))
        valid_size = len(remaining_indices) - train_size
        gen = torch.Generator().manual_seed(se)
        train_indices, valid_indices = random_split(remaining_indices, [train_size, valid_size], generator=gen)

        train_dataset = ResnetSubset(full_dataset, train_indices, transform=True)
        valid_dataset = ResnetSubset(full_dataset, valid_indices, transform=False)
        test_dataset = ResnetSubset(full_dataset, test_indices, transform=False)

        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
        class_weights = [1.0 / sum(np.array(train_labels) == i) for i in range(2)]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        if model_type == 'BiLSTM':
            model = BiLSTMModel(input_dim).to(device)
        elif model_type == 'ResNet32':
            model = ResNet32(input_dim).to(device)

        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_path = os.path.join(save_dir, f"{model_type}_model_seed{se}_fold{run_idx}.pth")
        txt_dir = os.path.join(save_dir, f"{model_type}_train_results_seed{se}_fold{run_idx}_results")
        fig_path = os.path.join(txt_dir, f"{model_type}_train_results_seed{se}_fold{run_idx}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, full_dataset, title=class_names[GT_class]
        )

        print(f"Seed {se} (Fold {run_idx}) Test F1: {f1:.4f}")
        all_f1_scores.append(f1)
        all_avg_times.append(avg_time_per_sample)
        all_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_results(model, input_dim, category_ratio, seeds, all_f1_scores, all_avg_times, all_acc, best_f1, best_seed, best_model_path, save_dir)
