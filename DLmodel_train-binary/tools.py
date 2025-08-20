import random
import numpy as np
import torch
import os, sys

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多張 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 計算 F1-score 的函數
def f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0  # 或 return np.nan 若你想保留 NaN 以作為後續辨識
    f1 = 2 * tp / denominator
    return f1

def compute_f1_score(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return f1_score(y_true, y_pred)  

def write_results(model, input_dim, category_ratio, seeds, all_f1_scores, all_sample_times, all_acc, best_f1, best_seed, best_model_path, save_dir):
    # 🔍 顯示結果 & 建立結果字串
    summary_lines = []
    summary_lines.append("\n✅ F1 scores from each seed:")
    for se, f1, st, ac in zip(seeds, all_f1_scores, all_sample_times, all_acc):
        summary_lines.append(f"Seed {se}: F1 = {f1:.4f}, Average Time per Sample = {st:.6f} seconds, Accuracy = {ac:.4f}")

    summary_lines.append(f"\n📊 Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
    summary_lines.append(f"🏆 Best F1: {best_f1:.4f} from Seed {best_seed}")
    summary_lines.append(f"📁 Best model saved at: {best_model_path}")

    # 印出結果到 terminal
    for line in summary_lines:
        print(line)

    # 📄 寫入 txt 檔案
    txt_output_path = os.path.join(save_dir, "results_summary.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total}\n")
        f.write(f"Trainable parameters: {trainable}\n")
        f.write(f"Input dimension: {input_dim}\n")
        f.write(f"Category ratio: {category_ratio}\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n✅ 寫入完成：{txt_output_path}")