import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

@torch.no_grad()
def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, full_dataset, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(txt_dir, exist_ok=True)

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_time = 0.0

    # 推論
    for inputs, labels, indices in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        start = time.time()
        logits = model(inputs)
        batch_loss = criterion(logits, labels.float())
        end = time.time()

        total_time += (end - start)
        total_loss += batch_loss.item()

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)              # [N, C]
    labels = torch.cat(all_labels, dim=0).int()        # [N, C]
    probs  = torch.sigmoid(logits)
    preds  = (probs > 0.5).int()

    # 整體多標籤指標（macro-F1 over classes 基於本回合）
    f1_macro = f1_score(labels.numpy(), preds.numpy(), average='macro', zero_division=0)

    # 這裡的「Accuracy」採用逐樣本逐類別的平均正確率
    acc = (preds.eq(labels)).float().mean().item()

    # 單樣本平均耗時
    avg_time_per_sample = total_time / max(1, logits.shape[0])

    # ====== 逐類別 TP/FP/FN/TN 與 per-class F1 ======
    per_class = {}
    f1_lines = ["Per-class F1 Scores:"]
    for c in range(num_classes):
        y_true = labels[:, c].numpy()
        y_pred = preds[:, c].numpy()

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        support_pos = int(np.sum(y_true == 1))

        denom = 2 * tp + fp + fn
        f1_c = (2 * tp / denom) if denom > 0 else 0.0

        per_class[c] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Support": support_pos, "F1": f1_c
        }
        f1_lines.append(f"Class {c}: {f1_c:.4f}")

    # 輸出 per-class F1（純文字）
    with open(os.path.join(txt_dir, "per_class_f1_scores.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f1_lines) + "\n")

    # 輸出 per-class 混淆計數（JSON）
    with open(os.path.join(txt_dir, "per_class_confusion.json"), "w", encoding="utf-8") as f:
        json.dump(per_class, f, ensure_ascii=False, indent=2)

    # 也輸出一次整體摘要
    with open(os.path.join(txt_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Samples: {logits.shape[0]}\n")
        f.write(f"Macro-F1: {f1_macro:.4f}\n")
        f.write(f"Accuracy(mean over labels): {acc:.4f}\n")
        f.write(f"Avg time per sample: {avg_time_per_sample:.6f} sec\n")

    # 回傳供 train 端記錄
    avg_loss = total_loss / max(1, len(test_loader))
    return avg_loss, f1_macro, avg_time_per_sample, acc
