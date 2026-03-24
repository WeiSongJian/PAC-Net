import os
import pandas as pd
import torch
import torch.nn as nn
from data_process import process
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils.utils import CSVLogger_my
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from antigen_antibody_emb import configuration, antibody_antigen_dataset
from antibinder_model import antibinder

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_targets(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--test_targets', type=str, default=None,
                        help='Comma-separated target names to test (e.g., "SARS-CoV2_Alpha"). If None, test all.')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='./test_results')

    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. 加载模型 ==========
    model = antibinder(
        antibody_hidden_dim=1024,
        antigen_hidden_dim=1024,
        latent_dim=args.latent_dim,
        res=False
    ).to(device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"✅ Loaded model from {args.checkpoint}")

    # ========== 2. 加载并筛选测试数据 ==========
    df = pd.read_csv(args.data_path)
    # 删除缺失关键字段的行
    df = df.dropna(subset=['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']).reset_index(drop=True)

    # 筛选测试样本（按 Target）
    if args.test_targets:
        target_list = [x.strip() for x in args.test_targets.split(',')]
        test_df = df[df['Target'].isin(target_list)].reset_index(drop=True)
    else:
        test_df = df

    test_idx = test_df.index.tolist()
    train_idx = []
    print("Running data preprocessing for test set...")
    metadata_path, _ = process(
        fold=0,
        train_idx=train_idx,  # 可为空列表 []
        val_idx=test_idx,  # 关键：包含你要测试的样本
        output_suffix="_test"
    )

    # ========== 3. 创建测试数据集和加载器 ==========
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)

    test_dataset = antibody_antigen_dataset(
        antigen_config=antigen_config,
        antibody_config=antibody_config,
        data_path=args.data_path,  # 注意：dataset 内部可能仍需要原始路径，但实际使用 df_test
        data=test_df,              # 假设 antibody_antigen_dataset 支持传入 DataFrame
        train=False,
        test=True,
        rate1=0.0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ========== 4. 推理 ==========
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for antibody_set, antigen_set, label in tqdm(test_loader, desc="Testing"):
            probs = model(antibody_set, antigen_set)  # shape: [B]
            probs = torch.sigmoid(probs)  # 转为概率（假设模型输出 logits）
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # ========== 5. 计算指标 ==========
    # 处理单类别情况
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 1:
        auc = None
        tn, fp, fn, tp = (len(all_labels), 0, 0, 0) if unique_labels[0] == 0 else (0, 0, 0, len(all_labels))
    else:
        auc = roc_auc_score(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # ========== 6. 保存结果 ==========
    log_file = os.path.join(
        args.output_dir,
        f"test_{'_'.join(parse_targets(args.test_targets)) if args.test_targets else 'all'}.csv"
    )
    logger = CSVLogger_my(
        ['auc', 'accuracy', 'precision', 'recall', 'f1', 'TN', 'FP', 'FN', 'TP'],
        log_file
    )
    logger.log([auc, acc, prec, rec, f1, tn, fp, fn, tp])

    # 打印结果
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"AUC       : {auc:.4f}" if auc is not None else "AUC       : N/A")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"\nResults saved to: {log_file}")