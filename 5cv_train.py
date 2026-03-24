import os
from sklearn.model_selection import KFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
import pandas as pd
import shutil
from antigen_antibody_emb import configuration, antibody_antigen_dataset
from model import PAC, AntiModelIinitial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import sys
import argparse
import json

from data_process import process
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
sys.path.append ('../')
import warnings
warnings.filterwarnings("ignore")


script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==================== Label Smoothing BCE ====================
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth, pos_weight=self.pos_weight)

class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, args, logger, load=False) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        self.best_loss = float('inf')
        self.best_val_metrics = None
        self.load = load
        if self.load == False:
            self.init()
        else:
            print("no init model")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        print("init successfully!")

    def matrix(self, yhat, y):
        return sum(y == yhat)

    def matrix_val(self, yhat, y):
        if len(y) == 0:
            return 0.0, 0.0, 0.0, 0.0
        yhat = np.array(yhat).flatten()
        y = np.array(y).flatten()
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            acc = accuracy_score(y, yhat)
            return acc, acc, acc, acc
        return (
            accuracy_score(y, yhat),
            precision_score(y, yhat, zero_division=0),
            f1_score(y, yhat, zero_division=0),
            recall_score(y, yhat, zero_division=0)
        )

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        num_samples = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for antibody_set, antigen_set, label, cdr_infos in dataloader:
                probs = self.model(antibody_set, antigen_set)
                y = label.float().to(device)

                loss = criterion(probs.view(-1), y.view(-1))
                total_loss += loss.item() * antibody_set[0].shape[0]
                num_samples += antibody_set[0].shape[0]

                yhat = (probs > 0.5).long()
                if yhat.is_cuda:
                    yhat = yhat.cpu()
                all_preds.append(yhat.numpy())
                if y.is_cuda:
                    y = y.cpu()
                all_labels.append(y.numpy())
                all_probs.append(torch.sigmoid(probs).cpu().numpy())

        all_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
        all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
        all_probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])

        acc, prec, f1, rec = self.matrix_val(all_preds, all_labels)

        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
            pr_auc = average_precision_score(all_labels, all_probs)
        else:
            pr_auc = 0.0

        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss, acc, prec, f1, rec, pr_auc

    def train(self, criterion, epochs=250):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.monitor_metric == 'loss':
            mode = 'min'
            monitor_key = 'val_loss'
        elif self.args.monitor_metric in ['f1', 'pr_auc']:
            mode = 'max'
            monitor_key = f'val_{self.args.monitor_metric}'
        else:
            raise ValueError(f"Unknown monitor_metric: {self.args.monitor_metric}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=25, mode=mode)
        best_val_loss = float('inf')
        self.best_val_metrics = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            num_train = 0
            all_train_preds = []
            all_train_labels = []
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for i, (antibody_set, antigen_set, label, cdr_infos) in enumerate(pbar):
                probs = self.model(antibody_set, antigen_set)
                y = label.float().to(device)

                loss = criterion(probs.view(-1), y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * antibody_set[0].shape[0]
                num_train += antibody_set[0].shape[0]

                yhat = (probs > 0.5).long().cpu().numpy()
                all_train_preds.append(yhat)
                all_train_labels.append(y.cpu().numpy())

                if i % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{np.mean(yhat == y.cpu().numpy()):.4f}"
                    })

            train_acc, train_prec, train_f1, train_rec = self.matrix_val(
                np.concatenate(all_train_preds),
                np.concatenate(all_train_labels)
            )
            train_loss = train_loss / num_train
            train_loss_exp = np.exp(train_loss)

            val_loss, val_acc, val_prec, val_f1, val_rec, val_pr_auc = self.evaluate(
                self.valid_dataloader, criterion
            )
            val_loss_exp = np.exp(val_loss)

            scheduler.step(val_loss if self.args.monitor_metric == 'loss' else
                          val_f1 if self.args.monitor_metric == 'f1' else val_pr_auc)

            current_metric = val_loss if self.args.monitor_metric == 'loss' else \
                           val_f1 if self.args.monitor_metric == 'f1' else val_pr_auc

            if early_stopping(current_metric):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            
            self.logger.log([
                epoch + 1,
                train_loss_exp, train_acc, train_prec, train_f1, train_rec,
                val_loss_exp, val_acc, val_prec, val_f1, val_rec, val_pr_auc
            ])

            if (self.args.monitor_metric == 'loss' and val_loss < best_val_loss) or \
               (self.args.monitor_metric != 'loss' and current_metric > getattr(self, 'best_metric', -float('inf'))):

                if self.args.monitor_metric == 'loss':
                    best_val_loss = val_loss
                else:
                    self.best_metric = current_metric

                self.best_val_metrics = {
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_prec': val_prec,
                    'val_rec': val_rec,
                    'val_pr_auc': val_pr_auc
                }

                fold_ckpt_dir = os.path.join("./ckpts", f"fold_{fold}")
                os.makedirs(fold_ckpt_dir, exist_ok=True)
                ckpt_path = f"./ckpts/{self.args.model_name}_fold{fold}_best.pth"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Epoch {epoch + 1}: Saved new best model ({monitor_key}={current_metric:.4f})")

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss_exp:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_exp:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} PR-AUC: {val_pr_auc:.4f}")

        print(f"\nFold completed. Best Val Metrics: {self.best_val_metrics}")
        return self.best_val_metrics


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        else:
            if self.mode == 'min':
                if val_metric < self.best_score - self.min_delta:
                    self.best_score = val_metric
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                if val_metric > self.best_score + self.min_delta:
                    self.best_score = val_metric
                    self.counter = 0
                else:
                    self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def custom_collate(batch):
    antibody_batch = []
    at_type_batch = []
    antibody_structure_batch = []
    antigen_batch = []
    antigen_structure_batch = []
    label_batch = []
    
    cdr_infos = []

    for item in batch:
        
        if len(item) == 4:
            ab_tuple, ag_tuple, label, cdr_info = item
            cdr_infos.append(cdr_info)
        else:
            raise ValueError(f"Unexpected item length in batch: {len(item)}")

        antibody, at_type, antibody_structure = ab_tuple
        antigen, antigen_structure = ag_tuple

        antibody_batch.append(antibody)
        at_type_batch.append(at_type)
        antibody_structure_batch.append(antibody_structure)
        antigen_batch.append(antigen)
        antigen_structure_batch.append(antigen_structure)
        label_batch.append(label)

    antibody_batch = torch.nn.utils.rnn.pad_sequence(antibody_batch, batch_first=True, padding_value=0)
    at_type_batch = torch.nn.utils.rnn.pad_sequence(at_type_batch, batch_first=True, padding_value=0)
    antibody_structure_batch = torch.stack(antibody_structure_batch)
    antigen_batch = torch.nn.utils.rnn.pad_sequence(antigen_batch, batch_first=True, padding_value=0)
    antigen_structure_batch = torch.stack(antigen_structure_batch).contiguous()
    label_batch = torch.stack(label_batch)

    
    return (
        [antibody_batch, at_type_batch, antibody_structure_batch],
        [antigen_batch, antigen_structure_batch],
        label_batch,
        cdr_infos  
    )


def backup_source_files(source_files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for src_file in source_files:
        src_path = os.path.join(script_dir, src_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, target_dir)
            print(f"Copied {src_file} to {target_dir}")
        else:
            print(f"Warning: {src_file} not found, skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--model_name', type=str, default='PAC')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data', type=str, default='train')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--data_path', type=str, default='./datasets/process_data/COVID-19/Cov_with_target_split.csv')

    
    parser.add_argument('--cv_strategy', type=str, default='kfold',
                        choices=['kfold', 'group_kfold', 'stratified_group_kfold'],
                        help='Cross-validation strategy')
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=['bce', 'weighted_bce', 'focal'],
                        help='Loss function type')
    parser.add_argument('--monitor_metric', type=str, default='loss',
                        choices=['loss', 'f1', 'pr_auc'],
                        help='Metric to monitor for early stopping and lr scheduler')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use WeightedRandomSampler for long-tail distribution')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0.0 to disable)')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"Using CUDA_VISIBLE_DEVICES={args.device}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: No CUDA devices detected!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',512)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',160)

    original_df = pd.read_csv(args.data_path)
    original_df = original_df.dropna(
        subset=['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
    ).reset_index(drop=True)

    df = original_df.copy()
    df['group_col'] = df['vh'].astype('category').cat.codes
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    
    if args.cv_strategy == 'kfold':
        cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_splits = cv.split(df)
    elif args.cv_strategy == 'group_kfold':
        cv = GroupKFold(n_splits=5)
        cv_splits = cv.split(df, groups=df['vh'])
    elif args.cv_strategy == 'stratified_group_kfold':
        if StratifiedGroupKFold is not None:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
            cv_splits = cv.split(df, df['ANT_Binding'], groups=df['vh'])
        else:
            print("StratifiedGroupKFold not available, falling back to GroupKFold")
            cv = GroupKFold(n_splits=5)
            cv_splits = cv.split(df, groups=df['vh'])
    else:
        raise ValueError(f"Unknown cv_strategy: {args.cv_strategy}")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, "global_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Global config saved to {config_path}")

    source_files_to_backup = [
        "model.py",
        "antigen_antibody_emb.py",
        "data_process.py",
        "5cv_train.py",
    ]

    fold_results = []
    print(f"Starting 5-fold cross-validation with strategy: {args.cv_strategy}")

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"=== Processing Fold {fold} ===")
        metadata_path_fold, processed_dir_fold = process(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            output_suffix=f"_fold{fold}"
        )

        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'=' * 50}")

        train_labels = df.iloc[train_idx]['ANT_Binding'].values
        val_labels = df.iloc[val_idx]['ANT_Binding'].values

        train_antibodies = df.iloc[train_idx]['vh'].unique()
        val_antibodies = df.iloc[val_idx]['vh'].unique()
        common_antibodies = set(train_antibodies) & set(val_antibodies)
        print(f"Train antibodies: {len(train_antibodies)}, Val antibodies: {len(val_antibodies)}")
        print(f"Common antibodies: {len(common_antibodies)} ({len(common_antibodies) / len(train_antibodies):.2%})")

        train_antigens = df.iloc[train_idx]['Antigen Sequence'].unique()
        val_antigens = df.iloc[val_idx]['Antigen Sequence'].unique()
        common_antigens = set(train_antigens) & set(val_antigens)
        print(f"Train antigens: {len(train_antigens)}, Val antigens: {len(val_antigens)}")
        print(f"Common antigens: {len(common_antigens)} ({len(common_antigens) / len(train_antigens):.2%})")

        print(f"Train set: Pos={train_labels.mean():.3f} ({np.sum(train_labels)} / {len(train_labels)})")
        print(f"Val set:   Pos={val_labels.mean():.3f} ({np.sum(val_labels)} / {len(val_labels)})")

        
        fold_log_dir = os.path.join(log_dir, f"fold_{fold}")
        os.makedirs(fold_log_dir, exist_ok=True)
        backup_source_files(source_files_to_backup, fold_log_dir)

        
        fold_config_path = os.path.join(fold_log_dir, "fold_config.json")
        with open(fold_config_path, "w") as f:
            fold_args = vars(args).copy()
            fold_args.update({
                'fold': fold,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'train_pos_ratio': float(train_labels.mean()),
                'val_pos_ratio': float(val_labels.mean()),
            })
            json.dump(fold_args, f, indent=4)

        df.iloc[train_idx].to_csv(os.path.join(fold_log_dir, "train.csv"), index=False)
        df.iloc[val_idx].to_csv(os.path.join(fold_log_dir, "val.csv"), index=False)

        
        train_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            metadata_path=metadata_path_fold,
            data=df.iloc[train_idx],
            train=True,
            test=False,
            rate1=1.0
        )

        val_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            metadata_path=metadata_path_fold,
            data=df.iloc[val_idx],
            train=False,
            test=True,
            rate1=0.0
        )

        
        sampler = None
        if args.use_weighted_sampler:
            antibody_counts = df['vh'].value_counts().to_dict()
            sample_weights = [1.0 / antibody_counts[df.iloc[i]['vh']] for i in train_idx]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,  
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )

        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )

        model = PAC(
            antibody_hidden_dim=1024,
            antigen_hidden_dim=1024,
            latent_dim=args.latent_dim,
            res=False
        ).to(device)

        log_file = os.path.join(fold_log_dir,
                                f"{args.model_name}_fold{fold}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
        
        logger = CSVLogger_my([
            'epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall',
            'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall', 'val_pr_auc'
        ], log_file)

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=val_dataloader,
            logger=logger,
            args=args,
            load=False
        )
        
        pos_count = np.sum(train_labels)
        neg_count = len(train_labels) - pos_count
        pos_weight_val = neg_count / (pos_count + 1e-8)

        if args.loss_type == 'bce':
            if args.label_smoothing > 0:
                criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=args.label_smoothing)
            else:
                criterion = nn.BCELoss()
        elif args.loss_type == 'weighted_bce':
            pos_weight = torch.tensor([pos_weight_val], device=device)
            if args.label_smoothing > 0:
                criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=args.label_smoothing, pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.loss_type == 'focal':
            if args.label_smoothing > 0:
                print("Warning: Focal Loss typically not used with label smoothing")
            criterion = FocalLoss(alpha=1, gamma=2)
        else:
            raise ValueError(f"Unknown loss_type: {args.loss_type}")

        best_metrics = trainer.train(criterion=criterion, epochs=args.epochs)
        # if fold == 0:  
        #     print(f"\nGenerating attention heatmap for fold {fold}...")
        # 
        #    
        #     ckpt_path = f"./ckpts/{args.model_name}_fold{fold}_best.pth"
        #     if os.path.exists(ckpt_path):
        #         model.load_state_dict(torch.load(ckpt_path, map_location=device))
        #         print(f"Loaded best model from {ckpt_path}")
        #     else:
        #         print("Best model checkpoint not found, using current model state.")

            model.eval()
        fold_results.append(best_metrics)

    
    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation Results")
    print("=" * 50)
    for i, res in enumerate(fold_results):
        if res:
            print(f"Fold {i + 1}: Val Acc={res['val_acc']:.4f}, F1={res['val_f1']:.4f}, PR-AUC={res['val_pr_auc']:.4f}")

    valid_results = [res for res in fold_results if res is not None]
    if valid_results:
        avg_metrics = {k: np.mean([res[k] for res in valid_results]) for k in valid_results[0]}
        print("\nAverage Metrics:")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

        with open(os.path.join(log_dir, "cv_results.txt"), "w") as f:
            f.write("5-Fold Cross-Validation Results\n")
            f.write(f"Params: {vars(args)}\n\n")
            for i, res in enumerate(valid_results):
                f.write(f"Fold {i + 1}: {res}\n")
            f.write("\nAverage Metrics:\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
    else:
        print("No valid results to average.")