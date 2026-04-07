
!pip install imodelsx -q

import os
import glob
import time
import random
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, auc, matthews_corrcoef, cohen_kappa_score
)

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Settings & reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

FIG_DIR = Path('./figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# XGB variants
XGB_VARIANTS = [
    {'eta':0.1,  'max_depth':6, 'subsample':0.8, 'colsample_bytree':0.8, 'seed': SEED},
    {'eta':0.05, 'max_depth':8, 'subsample':0.9, 'colsample_bytree':0.8, 'seed': SEED+11},
    {'eta':0.15, 'max_depth':5, 'subsample':0.8, 'colsample_bytree':0.9, 'seed': SEED+22},
]

# KAN hyperparams
KAN_HP = dict(
    hidden_layer_size=512,
    regularize_activation=0.0,
    regularize_entropy=0.0,
    regularize_ridge=0.2,
    spline_order=3,
    batch_size=32,
    lr=0.0025,
    weight_decay=0.0001,
    n_epochs=100
)

TEST_SIZE_TOTAL = 0.30
N_FOLDS = 5
VERBOSE = True

# internal_test_size to inject for older KAN.fit internals
internal_test_size = 0.15 / (0.70 + 0.15)

# Try import KANClassifier
KAN_AVAILABLE = True
try:
    from kan_sklearn import KANClassifier
    print("Imported KANClassifier from kan_sklearn")
except Exception:
    try:
        from imodelsx.kan.kan_sklearn import KANClassifier
        print("Imported KANClassifier from imodelsx.kan.kan_sklearn")
    except Exception:
        print("KANClassifier import failed; KAN will be disabled for this run.")
        KAN_AVAILABLE = False

import os
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

import warnings
import logging

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*datetime\.datetime\.utcnow.*",
    module=r".*jupyter_client.*"
)

warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy.*has no __module__ attribute"
)

warnings.filterwarnings(
    "ignore",
    message=r"Importing 'parser.split_arg_string' is deprecated"
)

warnings.filterwarnings("ignore", module=r"spacy\..*", category=DeprecationWarning)
warnings.filterwarnings("ignore", module=r"weasel\..*", category=DeprecationWarning)

for lg in ("jupyter_client", "ipykernel", "tornado", "asyncio", "spacy", "weasel"):
    try:
        logging.getLogger(lg).setLevel(logging.ERROR)
    except Exception:
        pass


# ---------------------------
# I/O & preprocessing utilities
# ---------------------------

def load_csv_flex():
    candidates = [
        '/content/All-samples-merged_deduped.csv',
        '/mnt/data/All-samples-merged_deduped.csv',
        '/content/All-samples-merged.csv',
        '/mnt/data/All-samples-merged.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            print("Loading dataset from:", p)
            return pd.read_csv(p)
    csvs = glob.glob('/content/*.csv') + glob.glob('*.csv') + glob.glob('/mnt/data/*.csv')
    if csvs:
        chosen = sorted(csvs)[0]
        print("Loading dataset from:", chosen)
        return pd.read_csv(chosen)
    raise FileNotFoundError("No CSV found. Place dataset in /content/ or /mnt/data/ or upload in Colab.")


def parse_family_subfamily(df, category_col='Category'):
    d = df.copy()
    if category_col in d.columns:
        parts = d[category_col].fillna('').astype(str).str.split('-', n=2, expand=True)
        d['Family'] = parts[0].str.strip().replace('', 'Unknown')
        d['Subfamily'] = parts[1].str.strip().fillna('')
    else:
        if 'Family' not in d.columns:
            d['Family'] = 'Unknown'
        if 'Subfamily' not in d.columns:
            d['Subfamily'] = ''
    d['Family'] = d['Family'].astype(str).str.strip().str.title()
    d['Subfamily'] = d['Subfamily'].astype(str).str.strip().str.title()
    return d


def detect_id_like_columns(df):
    id_patterns = ['id','sha','md5','hash','filename','idx','index','sample','guid']
    id_cols = [c for c in df.columns if any(p in c.lower() for p in id_patterns)]
    if id_cols:
        print("Note: id-like columns detected (no automatic dropping performed). Review for leakage if needed.")
        print("Flagged id-like columns:", id_cols)
    return id_cols


def handle_nans_only(df, fill_nan_with_zero=True):
    d = df.copy()
    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        nan_mask = d[numeric_cols].isna().any()
        if nan_mask.any():
            cols_with_nan = nan_mask[nan_mask].index.tolist()
            print("Numeric columns with NaNs:", cols_with_nan)
            if fill_nan_with_zero:
                print("Filling NaNs in numeric columns with 0.0")
                d[numeric_cols] = d[numeric_cols].fillna(0.0)
    return d


# ---------------------------
# Plot & metrics utilities
# ---------------------------

def plot_and_save_confusion(cm, classes, task_name):
    fig, ax = plt.subplots(figsize=(max(6,len(classes)*0.6), max(4,len(classes)*0.45)))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title(f'Confusion Matrix — {task_name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    thresh = cm.max() / 2.0 if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"{task_name.replace(' ', '_')}_confusion.png"
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print(f"Saved confusion matrix to: {out}")
    plt.show(); plt.close(fig)


def plot_and_save_roc(all_labels, all_probs, le, task_name):
    n_classes = all_probs.shape[1]
    y_true_oh = label_binarize(all_labels, classes=np.arange(n_classes))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:,1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0,1], [0,1], linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {task_name}')
        ax.legend(loc='lower right')
        out = FIG_DIR / f"{task_name.replace(' ', '_')}_roc.png"
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f"Saved ROC to: {out}")
        plt.show(); plt.close(fig)
        return roc_auc
    else:
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(n_classes):
            if y_true_oh[:, i].sum() == 0:
                fpr[i] = np.array([0.0, 1.0]); tpr[i] = np.array([0.0, 1.0]); roc_auc[i] = np.nan
            else:
                fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        valid = [v for v in roc_auc.values() if not np.isnan(v)]
        macro_auc = np.mean(valid) if valid else np.nan
        fig, ax = plt.subplots(figsize=(8,6))
        for i in range(n_classes):
            label = f"{le.classes_[i]} (AUC={roc_auc[i]:.4f})" if not np.isnan(roc_auc[i]) else f"{le.classes_[i]} (AUC=nan)"
            ax.plot(fpr[i], tpr[i], lw=2, label=label)
        ax.plot([0,1], [0,1], linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Multiclass ROC Curves — {task_name}')
        ax.legend(loc='lower right', fontsize='small')
        out = FIG_DIR / f"{task_name.replace(' ', '_')}_roc_multiclass.png"
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f"Saved multiclass ROC to: {out}")
        plt.show(); plt.close(fig)
        return macro_auc


def per_class_metrics_from_cm(cm):
    n = cm.shape[0]; total = cm.sum()
    res = {}
    for i in range(n):
        TP = cm[i,i]; FN = cm[i,:].sum() - TP
        FP = cm[:,i].sum() - TP; TN = total - (TP+FP+FN)
        TPR = TP / (TP+FN) if (TP+FN)>0 else 0.0
        TNR = TN / (TN+FP) if (TN+FP)>0 else 0.0
        FPR = FP / (FP+TN) if (FP+TN)>0 else 0.0
        FNR = FN / (FN+TP) if (FN+TP)>0 else 0.0
        Precision = TP / (TP+FP) if (TP+FP)>0 else 0.0
        Recall = TPR
        F1 = 2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0.0
        Balanced_Acc = 0.5*(TPR + TNR)
        res[i] = {'TP':int(TP),'FP':int(FP),'TN':int(TN),'FN':int(FN),
                  'Precision':Precision,'Recall':Recall,'F1':F1,
                  'TPR':TPR,'TNR':TNR,'FPR':FPR,'FNR':FNR,'BalancedAcc':Balanced_Acc}
    return res


def compute_metrics(y_true_arr, y_pred_arr, probs_arr, class_names):
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(class_names)))
    pc = per_class_metrics_from_cm(cm)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    try: mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
    except: mcc = np.nan
    try: kappa = cohen_kappa_score(y_true_arr, y_pred_arr)
    except: kappa = np.nan

    aucs_per_class = {}
    macro_auc = np.nan
    try:
        if probs_arr is not None:
            if probs_arr.shape[1] == 1:
                macro_auc = roc_auc_score(y_true_arr, probs_arr.ravel())
            else:
                y_bin = label_binarize(y_true_arr, classes=np.arange(len(class_names)))
                for i in range(len(class_names)):
                    try:
                        aucs_per_class[class_names[i]] = roc_auc_score(y_bin[:,i], probs_arr[:,i])
                    except Exception:
                        aucs_per_class[class_names[i]] = np.nan
                try:
                    macro_auc = roc_auc_score(y_bin, probs_arr, average='macro', multi_class='ovr')
                except Exception:
                    macro_auc = np.nan
    except Exception:
        macro_auc = np.nan

    TPR_mean = np.mean([pc[i]['TPR'] for i in pc])
    TNR_mean = np.mean([pc[i]['TNR'] for i in pc])
    FPR_mean = np.mean([pc[i]['FPR'] for i in pc])
    FNR_mean = np.mean([pc[i]['FNR'] for i in pc])
    balanced_acc = np.mean([pc[i]['BalancedAcc'] for i in pc])

    metrics = {
        'Accuracy': acc,
        'Precision_macro': prec_m,
        'Recall_macro': rec_m,
        'F1_macro': f1_m,
        'TPR_mean': TPR_mean,
        'TNR_mean': TNR_mean,
        'FPR_mean': FPR_mean,
        'FNR_mean': FNR_mean,
        'Balanced_Acc': balanced_acc,
        'MCC': mcc,
        'Kappa': kappa,
        'AUC_macro': macro_auc,
        'PerClass': pc,
        'PerClass_AUCs': aucs_per_class,
        'Confusion_Matrix': cm
    }
    return metrics


# ---------------------------
# utility: ensure probs shape (n_samples, n_classes)
# ---------------------------

def ensure_prob_matrix(pv, n_classes):
    pv = np.asarray(pv)
    if pv.ndim == 1:
        if n_classes == 2:
            pv = np.vstack([1 - pv, pv]).T
        else:
            pv = pv.reshape(-1,1)
    if pv.ndim == 2 and pv.shape[1] == 1 and n_classes == 2:
        pv = np.vstack([1 - pv.ravel(), pv.ravel()]).T
    if pv.ndim == 2 and pv.shape[1] != n_classes:
        # try to interpret as integer labels or softmax outputs of different shape
        if pv.dtype.kind in 'if' and pv.shape[1] < n_classes:
            preds = np.argmax(pv, axis=1)
        else:
            preds = pv.ravel().astype(int)
        oh = np.zeros((len(preds), n_classes))
        for i,p in enumerate(preds):
            if 0 <= int(p) < n_classes:
                oh[i,int(p)] = 1.0
        pv = oh
    return pv


def build_xgb_params(n_classes, seed):
    return {
        'objective': 'multi:softprob' if n_classes>2 else 'binary:logistic',
        'num_class': n_classes if n_classes>2 else None,
        'eval_metric': 'mlogloss' if n_classes>2 else 'logloss',
        'verbosity': 0,
        'seed': int(seed)
    }

# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == '__main__':
    df = load_csv_flex()
    print("Loaded shape:", df.shape)

    df = parse_family_subfamily(df, category_col='Category')
    detect_id_like_columns(df)
    if 'Category' in df.columns:
        unique_cats = df['Category'].dropna().astype(str).unique()
        print(f"Category column found: {len(unique_cats)} unique values (showing up to 10):", list(unique_cats)[:10])
        if any('gan' in s.lower() for s in unique_cats):
            print("Warning: 'GAN' variants detected in 'Category' values. Review if these are synthetic samples.")
    if 'Family' in df.columns:
        fam_vals = df['Family'].dropna().astype(str).unique()
        if any('gan' in s.lower() for s in fam_vals):
            print("Warning: 'GAN' variants detected in 'Family' values. Review if these are synthetic samples.")

    df = handle_nans_only(df, fill_nan_with_zero=True)

    TASKS = ['2-class','3-class','4-class','15-class','16-class']
    results = []

    for TASK in TASKS:
        print('\n' + '='*70)
        print('TASK:', TASK)
        d = df.copy()
        try:
            # selection logic (mirrors ANN notebook but robust)
            if TASK == '2-class':
                if 'Class' not in d.columns:
                    d['Class'] = np.where(d['Family'].astype(str).str.lower()=='benign', 'Benign', 'Malware')
                dft = d.copy(); target_col = 'Class'

            elif TASK == '3-class':
                wanted = ['Spyware','Ransomware','Trojan']
                present = [s.title() for s in d['Family'].astype(str).unique()]
                if set([w.title() for w in wanted]).issubset(set(present)):
                    dft = d[d['Family'].astype(str).str.title().isin(wanted)].copy()
                else:
                    nonben = d[d['Family'].astype(str).str.lower() != 'benign']
                    top3 = nonben['Family'].value_counts().index[:3].tolist()
                    print('3-class fallback using top-3 families:', top3)
                    dft = d[d['Family'].isin(top3)].copy()
                target_col = 'Family'

            elif TASK == '4-class':
                wanted = ['Benign','Spyware','Ransomware','Trojan']
                present = [s.title() for s in d['Family'].astype(str).unique()]
                if set([w.title() for w in wanted]).issubset(set(present)):
                    dft = d[d['Family'].astype(str).str.title().isin(wanted)].copy()
                else:
                    nonben = d[d['Family'].astype(str).str.lower() != 'benign']
                    top3 = nonben['Family'].value_counts().index[:3].tolist()
                    print('4-class fallback using Benign + top-3 families:', top3)
                    dft = d[d['Family'].isin(['Benign'] + top3)].copy()
                target_col = 'Family'

            elif TASK == '15-class':
                # robust selection: choose up to 15 non-Benign SUBFAMILIES with at least MIN_SAMPLES_PER_CLASS samples
                MIN_SAMPLES_PER_CLASS = 2
                nonben = d[d['Family'].astype(str).str.lower() != 'benign'].copy()
                sub_counts = nonben['Subfamily'].fillna('Unknown').astype(str).value_counts()
                selected = []
                for name, cnt in sub_counts.items():
                    if cnt >= MIN_SAMPLES_PER_CLASS:
                        selected.append(name)
                    if len(selected) >= 15:
                        break
                if len(selected) < 2:
                    print(f"Insufficient subfamilies meeting min samples ({MIN_SAMPLES_PER_CLASS}). Found: {len(selected)}")
                    raise ValueError('Too few stable classes for 15-class task.')
                print(f"Selected {len(selected)} subfamilies for 15-class (min_samples={MIN_SAMPLES_PER_CLASS})")
                dft = nonben[nonben['Subfamily'].astype(str).isin(selected)].copy()
                # include benign rows as class 'Benign' if present
                if 'Benign' in d['Family'].astype(str).unique():
                    benign_rows = d[d['Family'].astype(str).str.lower() == 'benign'].copy()
                    benign_rows['Subfamily'] = 'Benign'
                    dft = pd.concat([dft, benign_rows], axis=0, ignore_index=True)
                target_col = 'Subfamily'

            elif TASK == '16-class':
                d2 = d.copy()
                d2.loc[d2['Family'].astype(str).str.strip().str.lower() == 'benign', 'Subfamily'] = 'Benign'
                nonben = d2[d2['Family'].astype(str).str.lower() != 'benign'].copy()
                top15 = nonben['Subfamily'].value_counts().index[:15].tolist()
                benign = d2[d2['Family'].astype(str).str.lower() == 'benign'].copy()
                benign['Subfamily'] = 'Benign'
                sel_nonben = nonben[nonben['Subfamily'].isin(top15)].copy()
                dft = pd.concat([sel_nonben, benign], axis=0, ignore_index=True)
                target_col = 'Subfamily'

            else:
                raise ValueError('Unknown TASK: ' + str(TASK))
        except Exception as e:
            print('Selection failed for', TASK, 'err:', e)
            continue

        if dft.shape[0] < 5:
            print('Too few samples for task; skipping:', TASK)
            continue

        # prepare numeric features only (keep id-like columns in dataframe but modeling uses numeric columns)
        exclude = ['Category','Class','Family','Subfamily']
        features_df = dft.drop(columns=[c for c in exclude if c in dft.columns], errors='ignore').select_dtypes(include=[np.number]).copy()
        if features_df.shape[1] == 0:
            print('No numeric features for task', TASK, '— skipping.')
            continue
        features_df = features_df.fillna(0.0)
        X = features_df.values.astype(np.float32)
        y_names = dft[target_col].fillna('Unknown').values
        le = LabelEncoder().fit(y_names)
        y = le.transform(y_names)
        class_names = list(le.classes_)
        n_classes = len(class_names)
        print(f"Prepared: samples={len(y)}, features={features_df.shape[1]}, classes={class_names}")

        # check stratify viability
        vals, cnts = np.unique(y, return_counts=True)
        small = [(int(v), int(c)) for v,c in zip(vals,cnts) if c < 2]
        if small:
            print('Stratified split likely to fail — classes with <2 samples:', small)
            print('Skipping task due to insufficient class counts for stratification.')
            continue

        # 70/15/15 stratified split
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE_TOTAL, stratify=y, random_state=SEED)
        except Exception as e:
            print('Stratified split failed:', e); continue
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

        def counts_str(arr, name):
            vals, cnts = np.unique(arr, return_counts=True)
            return f"{name} counts: " + ", ".join(f"{int(v)}->{int(c)}" for v,c in zip(vals,cnts))
        print(counts_str(y, 'Full(encoded)'))
        print(counts_str(y_train, 'Train'))
        print(counts_str(y_val, 'Val'))
        print(counts_str(y_test, 'Test'))

        # standardize on train only
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # OOF stacking: deterministic blocks (XGBs then KAN)
        n_base_models = len(XGB_VARIANTS) + (1 if KAN_AVAILABLE else 0)
        n_train = X_train_s.shape[0]
        oof_meta = np.zeros((n_train, n_base_models * n_classes))

        # decide safe n_splits
        min_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
        n_splits = min(N_FOLDS, max(2, min_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        print('Starting OOF generation...')
        start_oof = time.time()
        fold_idx = 0
        for tr_idx, val_idx in skf.split(X_train_s, y_train):
            fold_idx += 1
            X_tr, X_val_fold = X_train_s[tr_idx], X_train_s[val_idx]
            y_tr, y_val_fold = y_train[tr_idx], y_train[val_idx]

            # XGB variants per fold
            for v_i, params in enumerate(XGB_VARIANTS):
                try:
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval   = xgb.DMatrix(X_val_fold, label=y_val_fold)
                    params_local = {
                        **build_xgb_params(n_classes, params.get('seed', SEED+v_i)),
                        'eta': params.get('eta',0.1),
                        'max_depth': params.get('max_depth',6),
                        'subsample': params.get('subsample',0.8),
                        'colsample_bytree': params.get('colsample_bytree',0.8)
                    }
                    params_local = {k:v for k,v in params_local.items() if v is not None}
                    bst = xgb.train(params_local, dtrain, num_boost_round=500, evals=[(dtrain,'train'),(dval,'val')],
                                    early_stopping_rounds=25, verbose_eval=False)
                    pv = bst.predict(dval)
                    pv = ensure_prob_matrix(pv, n_classes)
                    start_col = v_i * n_classes; end_col = (v_i+1) * n_classes
                    oof_meta[val_idx, start_col:end_col] = pv
                except Exception as e:
                    print(f"XGB variant {v_i} failed on fold {fold_idx}: {e}")
                    start_col = v_i * n_classes; end_col = (v_i+1) * n_classes
                    oof_meta[val_idx, start_col:end_col] = 0.0

            # KAN per-fold (if available) — robust: inject globals, try kwargs, fallback to no-kwargs
            if KAN_AVAILABLE:
                m_idx = len(XGB_VARIANTS)
                try:
                    kan_model = KANClassifier(
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        hidden_layer_size=KAN_HP.get('hidden_layer_size',512),
                        regularize_activation=KAN_HP.get('regularize_activation',0.0),
                        regularize_entropy=KAN_HP.get('regularize_entropy',0.0),
                        regularize_ridge=KAN_HP.get('regularize_ridge',0.0),
                        spline_order=KAN_HP.get('spline_order',3)
                    )
                    try:
                        kan_model.fit.__globals__['test_size'] = internal_test_size
                        kan_model.fit.__globals__['random_state'] = SEED
                        kan_model.fit.__globals__['shuffle'] = True
                    except Exception:
                        pass

                    sig = inspect.signature(kan_model.fit)
                    accepted = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
                    accepted -= {'self','X','y'}
                    fit_kwargs = {}
                    for k in ('batch_size','lr','weight_decay','n_epochs'):
                        if k in accepted and k in KAN_HP:
                            fit_kwargs[k] = KAN_HP[k]

                    try:
                        res = kan_model.fit(X_tr, y_tr, **fit_kwargs) if fit_kwargs else kan_model.fit(X_tr, y_tr)
                        if isinstance(res, tuple) and len(res) >= 1:
                            kan_model = res[0] or kan_model
                    except Exception as efit:
                        try:
                            kan_model.fit(X_tr, y_tr)
                        except Exception as e2:
                            raise RuntimeError(f"KAN.fit failed (fold): {efit}; {e2}")

                    try:
                        p_kan = kan_model.predict_proba(X_val_fold)
                    except Exception:
                        p_kan = None

                    if p_kan is None:
                        preds_kan = kan_model.predict(X_val_fold)
                        p_kan = np.zeros((len(preds_kan), n_classes))
                        for i,p in enumerate(preds_kan):
                            try:
                                idx = int(p)
                                if 0 <= idx < n_classes:
                                    p_kan[i, idx] = 1.0
                            except Exception:
                                pass

                    p_kan = ensure_prob_matrix(p_kan, n_classes)
                    start_col = m_idx * n_classes; end_col = (m_idx+1) * n_classes
                    oof_meta[val_idx, start_col:end_col] = p_kan
                except Exception as e:
                    print(f"KAN per-fold failed on fold {fold_idx}; filling zeros for KAN block. Err: {e}")
                    start_col = m_idx * n_classes; end_col = (m_idx+1) * n_classes
                    oof_meta[val_idx, start_col:end_col] = 0.0

            if VERBOSE:
                print(f" Fold {fold_idx} OOF filled; val_size={len(val_idx)}")

        print('OOF generation completed in %.1f s.' % (time.time() - start_oof))

        # train meta
        meta = LogisticRegression(max_iter=3000, multi_class='multinomial' if n_classes>2 else 'ovr', solver='lbfgs', C=1.0, random_state=SEED)
        meta.fit(oof_meta, y_train)
        print('Trained meta classifier on OOF meta-features.')

        # produce val meta-features (train each base on full TRAIN -> predict VAL)
        base_val_preds = []
        # XGB variants trained on full TRAIN
        dtrain_full_train = xgb.DMatrix(X_train_s, label=y_train)
        xgb_val_preds = []
        for v_i, params in enumerate(XGB_VARIANTS):
            params_local = {
                **build_xgb_params(n_classes, params.get('seed', SEED+v_i)),
                'eta': params.get('eta',0.1),
                'max_depth': params.get('max_depth',6),
                'subsample': params.get('subsample',0.8),
                'colsample_bytree': params.get('colsample_bytree',0.8)
            }
            params_local = {k:v for k,v in params_local.items() if v is not None}
            bst = xgb.train(params_local, dtrain_full_train, num_boost_round=500, verbose_eval=False)
            pv_val = bst.predict(xgb.DMatrix(X_val_s))
            pv_val = ensure_prob_matrix(pv_val, n_classes)
            xgb_val_preds.append(pv_val)
        xgb_val_avg = np.mean(np.stack(xgb_val_preds, axis=0), axis=0)
        base_val_preds.append(xgb_val_avg)

        # KAN trained on full TRAIN -> predict VAL
        if KAN_AVAILABLE:
            try:
                kan_full = KANClassifier(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    hidden_layer_size=KAN_HP.get('hidden_layer_size',512),
                    regularize_activation=KAN_HP.get('regularize_activation',0.0),
                    regularize_entropy=KAN_HP.get('regularize_entropy',0.0),
                    regularize_ridge=KAN_HP.get('regularize_ridge',0.0),
                    spline_order=KAN_HP.get('spline_order',3)
                )
                try:
                    kan_full.fit.__globals__['test_size'] = internal_test_size
                    kan_full.fit.__globals__['random_state'] = SEED
                    kan_full.fit.__globals__['shuffle'] = True
                except Exception:
                    pass

                sig = inspect.signature(kan_full.fit)
                accepted = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
                accepted -= {'self','X','y'}
                fit_kwargs = {}
                for k in ('batch_size','lr','weight_decay','n_epochs'):
                    if k in accepted and k in KAN_HP:
                        fit_kwargs[k] = KAN_HP[k]

                try:
                    res = kan_full.fit(X_train_s, y_train, **fit_kwargs) if fit_kwargs else kan_full.fit(X_train_s, y_train)
                    if isinstance(res, tuple) and len(res) >= 1:
                        kan_full = res[0] or kan_full
                except Exception:
                    kan_full.fit(X_train_s, y_train)

                try:
                    pv_kan_val = kan_full.predict_proba(X_val_s)
                except Exception:
                    pv_kan_val = None

                if pv_kan_val is None:
                    preds_kan = kan_full.predict(X_val_s)
                    pv_kan_val = np.zeros((len(preds_kan), n_classes))
                    for i,p in enumerate(preds_kan):
                        try:
                            pv_kan_val[i,int(p)] = 1.0
                        except Exception:
                            pass

                pv_kan_val = ensure_prob_matrix(pv_kan_val, n_classes)
                base_val_preds.append(pv_kan_val)
            except Exception as e:
                print("KAN train-on-full-train -> predict VAL failed; using zeros for KAN VAL block. Err:", e)
                base_val_preds.append(np.zeros((X_val_s.shape[0], n_classes)))

        # assemble meta_val and score
        meta_val = np.hstack(base_val_preds)
        if meta_val.shape[1] < oof_meta.shape[1]:
            pad = np.zeros((meta_val.shape[0], oof_meta.shape[1] - meta_val.shape[1]))
            meta_val = np.hstack([meta_val, pad])

        meta_val_preds = meta.predict(meta_val)
        meta_val_probs = meta.predict_proba(meta_val)

        val_metrics = compute_metrics(y_val, meta_val_preds, meta_val_probs, class_names)
        print('\nValidation metrics:')
        try:
            display(pd.DataFrame(val_metrics['Confusion_Matrix'], index=class_names, columns=class_names))
        except Exception:
            print(val_metrics['Confusion_Matrix'])
        for k in ['Accuracy','Precision_macro','Recall_macro','F1_macro','Balanced_Acc','MCC','Kappa','AUC_macro']:
            v = val_metrics.get(k, np.nan)
            print(f"  {k:20s}: {np.round(v,6) if isinstance(v,(float,np.floating)) else v}")

        # FINAL: train base models on train+val and predict TEST
        X_trainval_comb = np.vstack([X_train_s, X_val_s])
        y_trainval_comb = np.concatenate([y_train, y_val])

        test_preds_list = []
        dtrain_full = xgb.DMatrix(X_trainval_comb, label=y_trainval_comb)
        for v_i, params in enumerate(XGB_VARIANTS):
            params_local = {
                **build_xgb_params(n_classes, params.get('seed', SEED+v_i)),
                'eta': params.get('eta',0.1),
                'max_depth': params.get('max_depth',6),
                'subsample': params.get('subsample',0.8),
                'colsample_bytree': params.get('colsample_bytree',0.8)
            }
            params_local = {k:v for k,v in params_local.items() if v is not None}
            bst = xgb.train(params_local, dtrain_full, num_boost_round=500, verbose_eval=False)
            pv_test = bst.predict(xgb.DMatrix(X_test_s))
            pv_test = ensure_prob_matrix(pv_test, n_classes)
            test_preds_list.append(pv_test)
        xgb_test_avg = np.mean(np.stack(test_preds_list, axis=0), axis=0)
        base_test_preds = [xgb_test_avg]

        # KAN on train+val -> predict test
        if KAN_AVAILABLE:
            try:
                kan_full_tv = KANClassifier(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    hidden_layer_size=KAN_HP.get('hidden_layer_size',512),
                    regularize_activation=KAN_HP.get('regularize_activation',0.0),
                    regularize_entropy=KAN_HP.get('regularize_entropy',0.0),
                    regularize_ridge=KAN_HP.get('regularize_ridge',0.0),
                    spline_order=KAN_HP.get('spline_order',3)
                )
                try:
                    kan_full_tv.fit.__globals__['test_size'] = internal_test_size
                    kan_full_tv.fit.__globals__['random_state'] = SEED
                    kan_full_tv.fit.__globals__['shuffle'] = True
                except Exception:
                    pass

                sig = inspect.signature(kan_full_tv.fit)
                accepted = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
                accepted -= {'self','X','y'}
                fit_kwargs = {}
                for k in ('batch_size','lr','weight_decay','n_epochs'):
                    if k in accepted and k in KAN_HP:
                        fit_kwargs[k] = KAN_HP[k]
                try:
                    res = kan_full_tv.fit(X_trainval_comb, y_trainval_comb, **fit_kwargs) if fit_kwargs else kan_full_tv.fit(X_trainval_comb, y_trainval_comb)
                    if isinstance(res, tuple) and len(res) >= 1:
                        kan_full_tv = res[0] or kan_full_tv
                except Exception:
                    kan_full_tv.fit(X_trainval_comb, y_trainval_comb)

                try:
                    pv_kan_test = kan_full_tv.predict_proba(X_test_s)
                except Exception:
                    pv_kan_test = None

                if pv_kan_test is None:
                    preds_kan = kan_full_tv.predict(X_test_s)
                    pv_kan_test = np.zeros((len(preds_kan), n_classes))
                    for i,p in enumerate(preds_kan):
                        try:
                            pv_kan_test[i,int(p)] = 1.0
                        except Exception:
                            pass

                pv_kan_test = ensure_prob_matrix(pv_kan_test, n_classes)
                base_test_preds.append(pv_kan_test)
            except Exception as e:
                print("KAN train-on-train+val -> predict TEST failed; using zeros for KAN TEST block. Err:", e)
                base_test_preds.append(np.zeros((X_test_s.shape[0], n_classes)))

        # assemble meta_test
        meta_test = np.hstack(base_test_preds)
        if meta_test.shape[1] < oof_meta.shape[1]:
            pad = np.zeros((meta_test.shape[0], oof_meta.shape[1] - meta_test.shape[1]))
            meta_test = np.hstack([meta_test, pad])

        meta_test_preds = meta.predict(meta_test)
        meta_test_probs = meta.predict_proba(meta_test)

        test_metrics = compute_metrics(y_test, meta_test_preds, meta_test_probs, class_names)
        print('\nTest metrics:')
        try:
            display(pd.DataFrame(test_metrics['Confusion_Matrix'], index=class_names, columns=class_names))
        except Exception:
            print(test_metrics['Confusion_Matrix'])
        for k in ['Accuracy','Precision_macro','Recall_macro','F1_macro','Balanced_Acc','MCC','Kappa','AUC_macro']:
            v = test_metrics.get(k, np.nan)
            print(f"  {k:20s}: {np.round(v,6) if isinstance(v,(float,np.floating)) else v}")

        # per-class table
        print('\nPer-class metrics (test):')
        pc = test_metrics['PerClass']
        rows = []
        for i, cname in enumerate(class_names):
            r = pc[i]
            rows.append([cname, r['TP'], r['FP'], r['FN'], r['TN'], np.round(r['Precision'],4), np.round(r['Recall'],4), np.round(r['F1'],4), np.round(r['BalancedAcc'],4)])
        df_pc = pd.DataFrame(rows, columns=['class','TP','FP','FN','TN','Precision','Recall','F1','BalancedAcc'])
        display(df_pc)

        # Save & show figures (confusion + ROC)
        plot_and_save_confusion(test_metrics['Confusion_Matrix'], class_names, TASK)
        if meta_test_probs is not None:
            try:
                auc_val = plot_and_save_roc(y_test, meta_test_probs, le, TASK)
                print(f"AUC (from plotted curves) for {TASK}: {auc_val}")
            except Exception as e:
                print("ROC plotting failed:", e)

        results.append({
            'Task': TASK,
            'Model_combo': 'XGB_variants + KAN' if KAN_AVAILABLE else 'XGB_variants',
            'n_samples': len(y),
            'n_features': features_df.shape[1],
            'n_classes': n_classes,
            'Test_Accuracy': test_metrics['Accuracy'],
            'Test_F1_macro': test_metrics['F1_macro'],
            'Test_AUC_macro': test_metrics['AUC_macro']
        })

    # final summary
    if results:
        df_res = pd.DataFrame(results)
        print('\n' + '='*70)
        print('Summary (all tasks):')
        display(df_res)
    else:
        print('No completed tasks.')


