import torch
import time
from tools import set_seed, f1_score, write_results
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from dataset import *

def get_group_map(dataset):
    group_map = {}
    for idx, path in enumerate(dataset.sample_paths):
        parts = path.split(os.sep)
        group_name = next((p for p in parts if p.startswith("recording_")), None)
        if group_name:
            group_map.setdefault(group_name, []).append(idx)
    return group_map

def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, full_dataset, title = 'Confusion Matrix'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(save_path)
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()

    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []

    false_positives = []
    false_negatives = []

    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            for i in range(len(inputs)):
                sample_idx = indices[i].item()
                detailed_path = full_dataset.get_sample_path(sample_idx)

                if predicted[i] == 1 and labels[i] == 0:
                    false_positives.append(f"{str(detailed_path)}")
                elif predicted[i] == 0 and labels[i] == 1:
                    false_negatives.append(f"{str(detailed_path)}")

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred) 

    with open(f"{txt_dir}/false_positives.txt", "w") as fp_file:
        fp_file.write("\n".join(false_positives))

    with open(f"{txt_dir}/false_negatives.txt", "w") as fn_file:
        fn_file.write("\n".join(false_negatives))

    print(f"共有 {len(false_positives)} FP，{len(false_negatives)} FN")
    print(f"已保存到 {txt_dir}/false_positives.txt 和 false_negatives.txt")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()

    return avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_class',type=int)
    parser.add_argument('--SHAP',type=str, default=None)
    parser.add_argument('--F_type',type=str)
    parser.add_argument('--model', type=str, default='Resnet32', choices=['ResNet32', 'BiLSTM'])
    parser.add_argument('--data',type=str)
    parser.add_argument('--sport', type=str, default='benchpress', choices=['benchpress', 'deadlift'])
    args = parser.parse_args()

    GT_class = args.GT_class
    SHAP_mode = args.SHAP
    F_type = args.F_type
    model_type = args.model
    data = args.data
    sport = args.sport

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SHAP_mode is None:
        if F_type == '2D':
            if sport == 'deadlift':
                data_path = os.path.join(os.getcwd(), 'data', data)
                full_dataset = Dataset_dd2voz(data_path, GT_class)
                save_dir = os.path.join(os.getcwd(), 'models', 'deadlift', model_type,data, str(GT_class))
                category_ratio = full_dataset.get_ratio()
                P_ratio = category_ratio[str(GT_class)]

            if sport == 'benchpress':
                class_names = {0: 'tilting_to_the_left', 1: 'tilting_to_the_right', 2: 'elbows_flaring', 3: 'scapular_protraction'}
                data_path = os.path.join(os.getcwd(), 'data', data, 'bench_press_multilabel_cut4.csv')
                full_dataset = Dataset_Benchpress(data_path, GT_class)
                save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data, class_names[GT_class])
                os.makedirs(save_dir, exist_ok=True)
                category_ratio = full_dataset.get_ratio()
                P_ratio = category_ratio[1]

        elif F_type == '3D':
            data_path = os.path.join(os.getcwd(), 'data', '3D_Final')
            full_dataset = Dataset_3D(data_path, GT_class)
            save_dir = os.path.join(os.getcwd(), 'models', '3D_Final_Resnet32', str(GT_class))
        input_dim = full_dataset.dim
        print('input_dim', input_dim)

    print(f'Category : {category_ratio}')

    best_f1 = -1
    best_seed = None
    best_model_path = ""
    all_f1_scores = []
    all_avg_times = []
    all_acc = []
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        set_seed(se)

        group_map = get_group_map(full_dataset)
        all_group_names = list(group_map.keys())
        random.seed(se)
        random.shuffle(all_group_names)
        test_group_names = set(all_group_names[:16])

        print(f"\n📦 Seed {se} - Selected Test Groups:")
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

        test_dataset = ResnetSubset(full_dataset, test_indices, transform=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        if model_type == 'BiLSTM':
            model = BiLSTMModel(input_dim).to(device)
        elif model_type == 'Resnet32':
            model = ResNet32(input_dim).to(device)

        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))

        save_path = os.path.join(save_dir, f"{model_type}_model_seed{se}.pth")
        txt_dir = os.path.join(save_dir, f"{model_type}_train_results_seed{se}_results")
        fig_path = os.path.join(txt_dir, f"{model_type}_train_results_seed{se}.png")
        os.makedirs(txt_dir, exist_ok=True)

        avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, full_dataset, title='Confusion Matrix'
        )

        print(f"Seed {se} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)
        all_avg_times.append(avg_time_per_sample)
        all_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_results(model, input_dim, category_ratio, seeds, all_f1_scores, all_avg_times, all_acc, best_f1, best_seed, best_model_path, save_dir)
