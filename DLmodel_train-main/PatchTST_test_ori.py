import os
import torch
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import json
import matplotlib.pyplot as plt
import numpy as np
from tools import *
import argparse
from torch.utils.data import DataLoader, random_split
from models import PatchTSTClassifier
from sklearn.metrics import accuracy_score


def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, full_dataset, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    # **存放測試過程的數據**
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    cm_details = {str(i): [] for i in range(num_classes * num_classes)}
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.ndim != 3:
                raise ValueError(f"Expected 3D input (B, T, F), got shape {inputs.shape}")

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]

            for i in range(len(inputs)):
                sample_idx = indices[i].item()
                detailed_path = full_dataset.get_sample_path(sample_idx)
                
                # labels/preds shape: (B, 4)
                true_vec = labels[i].cpu().numpy()
                pred_vec = preds[i].cpu().numpy()
                
                if true_vec.sum() == 0 and pred_vec.sum() == 0:
                    continue  # 忽略全0樣本（屬於Category_0）
            
                # 找出 true_label 和 pred_label
                true_label = np.argmax(true_vec)
                pred_label = np.argmax(pred_vec)
                confidence_list  = probs[i].cpu().numpy().tolist()
                cm_index = true_label * num_classes + pred_label
                cm_details[str(cm_index)].append([
                    str(detailed_path), 
                    [float(c) for c in confidence_list]
                ])
                    
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 繪製混淆矩陣
    # binary_classes = ['left-side low', 'right-side low', 'shoulder press', 'shoulder up']
    # classes = ['Correct', 'left-side low', 'right-side low', 'shoulder press', 'shoulder up']
    # binary_classes = ['wrist press', 'left-side low', 'right-side low', 'shoulder press', 'shoulder up']
    # classes = ['Correct', 'wrist press', 'left-side low', 'right-side low', 'shoulder press', 'shoulder up']
    binary_classes = ['The barbell is moving away from the shins.', 'Hips rise before the barbell leaves the ground.', 'The barbell collides with the knees.', 'Lower back rounding']
    classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    
    cm = multilabel_confusion_matrix(y_true, y_pred, sample_weight=None, labels=None, samplewise=False)
    n_classes = cm.shape[0]
    fig, axes = plt.subplots(nrows=(n_classes+1)//2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=['False', 'True'])
        disp.plot(include_values=True, cmap="Blues", ax=axes[i], 
                xticks_rotation="horizontal", values_format="d")
        axes[i].set_title(f'Class: {binary_classes[i]}')

    plt.tight_layout()
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()
    
    cm = multilabel_confusion_matrix_mix(y_true, y_pred, num_classes)
    plot_custom_confusion_matrix(cm, classes, f"{txt_dir}/confusion_matrix_mix.png")
    with open(f"{txt_dir}/confusion_matrix_detail_paths.json", "w", encoding="utf-8") as f:
        json.dump(cm_details, f, indent=2, ensure_ascii=False)
    accuracy = accuracy_score(y_true, y_pred)    
        # ➤ 計算每個類別的 F1-score
    per_class_f1 = f1_score(y_true, y_pred, average=None)  # 這會回傳一個長度 = num_classes 的 list

    # ➤ 儲存 F1 到 TXT 檔
    f1_txt_path = os.path.join(txt_dir, "per_class_f1_scores.txt")
    with open(f1_txt_path, "w") as f:
        f.write("Per-class F1 Scores:\n")
        for i, f1_value in enumerate(per_class_f1):
            f.write(f"Class {i}: {f1_value:.4f}\n")


    return avg_loss, f1, avg_time_per_sample, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    args = parser.parse_args()
    
    from dataset.PatchTST import *
    if args.sport == 'deadlift':
        dataset = os.path.join(os.getcwd(), 'data', '3D_Real_Final')
        full_dataset = Dataset_TST_Deadlift(dataset)
        save_dir = f'./models/TST_Deadlift/12'
        num_classes = 4
        input_len = 110
        
    elif args.sport == 'benchpress':
        dataset = os.path.join(os.getcwd(), 'data', 'BPdata', 'bench_press_multilabel_cut4.csv')
        full_dataset = Dataset_TST_Benchpress(dataset)
        save_dir = f'./models/TST_Benchpress/2'
        num_classes = 4
        input_len = 100
        
    input_dim = full_dataset.dim
    print('Input dimention',input_dim)
    
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    seeds = [42, 2023, 7, 88, 100, 999,1717,3030]
    
    for se in seeds:
        set_seed(se)

        # 分割資料
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        train_indices, valid_indices, test_indices = random_split(
            range(len(full_dataset)), [train_size, valid_size, test_size],
            generator=gen
        )
        
        train_dataset = TransformSubset(full_dataset, train_indices, transform=True)
        valid_dataset = TransformSubset(full_dataset, valid_indices, transform=False)
        test_dataset  = TransformSubset(full_dataset, test_indices, transform=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        fig_path = os.path.join(save_dir, f"PatchTST_train_results_seed{se}.png")
        
        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, save_dir, save_path, full_dataset, num_classes
        )
        print(f"Seed {se} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_result(model, seeds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path)