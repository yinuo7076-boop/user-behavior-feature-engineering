import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder 
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from imblearn.over_sampling import SMOTE

# 数据路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# 读取数据
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'val.csv'))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))

# 删除不作为特征的列（注意：不再删除 behavior_type！）
drop_cols = [
    'user_id', 
    'item_id', 
    'time', 
    'label', 
    'total_buy',
    'buy_rate_x',
    'buy_count_x',
    'buy_rate_y',
    'buy_count_y',
    'buy_rate',
    'buy_recency',
    'buy_recency_hours',
    'repurchase_index',
    'impulsive_buying_rate',
    'hesitation_buying_rate'
    ]  
X_train = train_df.drop(columns=drop_cols, errors='ignore')
y_train = train_df['label']
X_val = val_df.drop(columns=drop_cols, errors='ignore')
y_val = val_df['label']
X_test = test_df.drop(
    columns=[col for col in drop_cols if col in test_df.columns],
    errors='ignore'
)

# 处理 test_df
if 'label' in test_df.columns:
    y_test = test_df['label']
else:
    y_test = None
X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])

# Step 1：去除无效特征
def filter_low_variance_and_missing(X, variance_threshold=0.01, missing_ratio_threshold=0.5):
    # 缺失率过滤
    missing_ratio = X.isnull().mean()
    valid_by_missing = missing_ratio[missing_ratio < missing_ratio_threshold].index
    X = X[valid_by_missing]
    # 方差过滤（仅对数值型）
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        vt = VarianceThreshold(threshold=variance_threshold)
        vt.fit(X[numeric_cols])
        kept_numeric = numeric_cols[vt.get_support()]
        # 合并非数值列（如类别型，通常不做方差过滤）
        non_numeric_cols = X.columns.difference(numeric_cols)
        final_cols = list(kept_numeric) + list(non_numeric_cols)
        X = X[final_cols]
    return X
X_train_step1 = filter_low_variance_and_missing(X_train)
# 对验证集同步保留相同列（避免泄露）
X_val_step1 = X_val[X_train_step1.columns]

# Step 2：消除冗余
def remove_high_correlation(X, threshold=0.8, business_priority=None):
    """
    X: DataFrame (仅数值特征)
    business_priority: list，业务上更重要的特征排在前面，优先保留
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] >= threshold].index.tolist()
        for hc in high_corr:
            # 保留业务价值更高的
            if business_priority is not None:
                if col in business_priority and hc not in business_priority:
                    to_drop.add(hc)
                elif hc in business_priority and col not in business_priority:
                    to_drop.add(col)
                else:
                    # 都在或都不在，保留第一个（或按顺序）
                    to_drop.add(hc)  # 默认保留 col，删 hc
            else:
                to_drop.add(hc)
    
    final_cols = [c for c in X.columns if c not in to_drop]
    return X[final_cols]

X_train_step2 = remove_high_correlation(X_train_step1, threshold=0.8)
X_val_step2 = X_val_step1[X_train_step2.columns]

# Step 3：排序 & 编码
cat_features = X_train_step2.select_dtypes(exclude=[np.number]).columns.tolist()
if cat_features:
    for col in cat_features:
        le = LabelEncoder()
        X_train_step2[col] = le.fit_transform(X_train_step2[col].astype(str))
        X_val_step2[col] = le.transform(X_val_step2[col].astype(str))

# 训练初始模型（全特征）
clf_initial = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    num_leaves=31,
    learning_rate=0.1,
    min_data_in_leaf=20,
    n_estimators=100,
    random_state=42,
    verbosity=-1
)

# X_train_step2 = X_train_step2[['item_category']]
# X_val_step2 = X_val_step2[['item_category']]

clf_initial.fit(
    X_train_step2, y_train,
    eval_set=[(X_val_step2, y_val)],
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10, verbose=False),
        log_evaluation(False)
    ]
)

# 基于重要性筛选特征
importance_df = pd.DataFrame({
    'feature': X_train_step2.columns,
    'importance': clf_initial.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
selected_features = importance_df[importance_df['cumulative_importance'] <= 0.95]['feature'].tolist()
if len(selected_features) == 0 and len(importance_df) > 0:
    selected_features = [importance_df.iloc[0]['feature']]

X_train_final = X_train_step2[selected_features]
X_val_final = X_val_step2[selected_features]
# y_val 保持原始索引

# 训练最终模型并评估
clf_final = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    num_leaves=31,
    learning_rate=0.1,
    min_data_in_leaf=20,
    n_estimators=100,
    random_state=42,
    verbosity=-1
)

clf_final.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10, verbose=False),
        log_evaluation(False)
    ]
)

# 排列重要性
perm_imp = permutation_importance(
    clf_final, X_val_final, y_val,
    n_repeats=5,
    random_state=42,
    scoring='neg_log_loss'
)
perm_df = pd.DataFrame({
    'feature': selected_features,
    'perm_importance_mean': perm_imp.importances_mean
}).sort_values('perm_importance_mean', ascending=False)

# 输出 & 保存
# print(f"原始特征数: {X_train.shape[1]}")
# print(f"步骤1后: {X_train_step1.shape[1]}")
# print(f"步骤2后: {X_train_step2.shape[1]}")
# print(f"最终保留: {len(selected_features)}")

leakage_features = [
    'behavior_type',
    'buy_recency_hours',
    'total_buy',
    'total_cart',
    'last_1day_pv',
    'last7day_pv'
]

selected_features = [
    f for f in selected_features
    if f not in leakage_features
]

X_train_final.to_csv(os.path.join(PROCESSED_DIR, 'X_train_selected.csv'),index=False)
X_val_final.to_csv(os.path.join(PROCESSED_DIR, 'X_val_selected.csv'), index=False)
pd.Series(selected_features).to_csv(os.path.join(PROCESSED_DIR, 'selected_features.txt'), index=False, header=False)

# Baseline Experiment
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
experiment_name = 'baseline'
EXPERIMENT_DIR = os.path.join(
    EXPERIMENTS_DIR,
    experiment_name
)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
# 用于记录训练过程
evals_result = {}

clf_baseline = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    num_leaves=31,
    learning_rate=0.1,
    min_data_in_leaf=20,
    n_estimators=100,
    random_state=42,
    verbosity=-1
)

clf_baseline.fit(
    X_train_final,
    y_train,
    eval_set=[(X_train_final, y_train), (X_val_final, y_val)],
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10, verbose=False),
        record_evaluation(evals_result),
        log_evaluation(False)
    ]
)

# 画学习曲线
plt.figure(figsize=(8,5))

plt.plot(
    evals_result['training']['binary_logloss'],
    label='Training Loss',
    linewidth=2
)

plt.plot(
    evals_result['valid_1']['binary_logloss'],
    label='Validation Loss',
    linewidth=2
)

plt.xlabel("Boosting Iterations")
plt.ylabel("Binary Log Loss")
plt.title("Learning Curve of Baseline LightGBM")
plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "experiments/baseline/learning_curve.png",
    dpi=300
)

plt.show()

# Predict
y_pred = clf_baseline.predict(X_val_final)
y_prob = clf_baseline.predict_proba(X_val_final)[:, 1]

# 评估指标
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f'[Baseline] Recall: {recall:.4f}')
print(f'[Baseline] F1: {f1:.4f}')
print(f'[Baseline] AUC: {auc:.4f}')

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Recall', 'F1', 'AUC'],
    'Value': [recall, f1, auc]
})

metrics_df.to_csv(
    os.path.join(EXPERIMENT_DIR, 'metrics.csv'),
    index=False
)

# Save roc cruve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(
    os.path.join(EXPERIMENT_DIR, 'roc_curve.png'),
    bbox_inches='tight'
)
plt.close()

# Save confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax)
plt.title('Confusion Matrix')
plt.savefig(
    os.path.join(EXPERIMENT_DIR, 'confusion_matrix.png'),
    bbox_inches='tight'
)
plt.close()

# Save model
joblib.dump(
    clf_baseline,
    os.path.join(EXPERIMENT_DIR, 'model.pkl')
)

# SMOTE 1.0/0.3/0.5/0.8 Experiment
smote_ratios = [1.0, 0.3, 0.5, 0.8]
for smote_ratio in smote_ratios:
    experiment_name = f'smote_{smote_ratio:.1f}'
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # Step 1: 对筛选后的训练集应用 SMOTE（只对训练集做！验证集保持原始分布）
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)
    print(f"[SMOTE {smote_ratio:.1f}] SMOTE 后训练集分布: {np.bincount(y_train_smote)}")  # 查看平衡效果
    # Step 2: 训练新模型（使用平衡后的数据）
    clf_smote = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=31,
        learning_rate=0.1,
        min_data_in_leaf=20,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    clf_smote.fit(
        X_train_smote, y_train_smote,  # ← 使用 SMOTE 后的数据
        eval_set=[(X_val_final, y_val)],  # ← 验证集仍用原始数据！
        eval_metric='binary_logloss',
        callbacks=[
            early_stopping(stopping_rounds=10, verbose=False),
            log_evaluation(False)
        ]
    )
    # Step 3: 预测与评估（在原始验证集上！）
    y_pred_smote = clf_smote.predict(X_val_final)
    y_prob_smote = clf_smote.predict_proba(X_val_final)[:, 1]

    recall_smote = recall_score(y_val, y_pred_smote)
    f1_smote = f1_score(y_val, y_pred_smote)
    auc_smote = roc_auc_score(y_val, y_prob_smote)

    print(f'[SMOTE {smote_ratio:.1f}] Recall: {recall_smote:.4f}')
    print(f'[SMOTE {smote_ratio:.1f}] F1: {f1_smote:.4f}')
    print(f'[SMOTE {smote_ratio:.1f}] AUC: {auc_smote:.4f}')

    # Step 4: 保存结果
    metrics_df_smote = pd.DataFrame({
        'Metric': ['Recall', 'F1', 'AUC'],
        'Value': [recall_smote, f1_smote, auc_smote]
    })
    metrics_df_smote.to_csv(
        os.path.join(EXPERIMENT_DIR, 'metrics.csv'),
        index=False
    )

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob_smote)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_smote:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (SMOTE {smote_ratio:.1f})')
    plt.legend()
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'roc_curve.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save Confusion Matrix
    cm_smote = confusion_matrix(y_val, y_pred_smote)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_smote)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix (SMOTE {smote_ratio:.1f})')
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'confusion_matrix.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save model
    joblib.dump(
        clf_smote,
        os.path.join(EXPERIMENT_DIR, 'model.pkl')
    )

    # Save SMOTE 后的训练集
    pd.DataFrame(X_train_smote, columns=selected_features).to_csv(
        os.path.join(EXPERIMENT_DIR, 'X_train_smote.csv'), index=False
    )
    pd.Series(y_train_smote).to_csv(
        os.path.join(EXPERIMENT_DIR, 'y_train_smote.csv'), index=False, header=['label']
    )

# Oversampling 1.0/0.3/0.5/0.8 Experiment    
oversampling_ratios = [1.0, 0.3, 0.5, 0.8]
for oversampling_ratio in oversampling_ratios:
    experiment_name = f'oversampling_{oversampling_ratio:.1f}'
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # Step 1: 对筛选后的训练集应用 Oversampling（只对训练集做！验证集保持原始分布）
    # 计算少数类样本数量
    minority_class_count = np.sum(y_train == 1)
    majority_class_count = np.sum(y_train == 0)
    desired_minority_count = int(majority_class_count * oversampling_ratio)

    if desired_minority_count > minority_class_count:
        # 过采样少数类
        oversample_indices = np.random.choice(
            np.where(y_train == 1)[0],
            size=desired_minority_count - minority_class_count,
            replace=True
        )
        X_train_oversampled = pd.concat([X_train_final, X_train_final.iloc[oversample_indices]], axis=0)
        y_train_oversampled = pd.concat([y_train, y_train.iloc[oversample_indices]], axis=0)
    else:
        X_train_oversampled = X_train_final.copy()
        y_train_oversampled = y_train.copy()

    print(f"[Oversampling {oversampling_ratio:.1f}] Oversampling 后训练集分布: {np.bincount(y_train_oversampled)}")  # 查看平衡效果

    # Step 2: 训练新模型（使用平衡后的数据）
    clf_oversample = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=31,
        learning_rate=0.1,
        min_data_in_leaf=20,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    clf_oversample.fit(
        X_train_oversampled, y_train_oversampled,  # ← 使用 Oversampling 后的数据
        eval_set=[(X_val_final, y_val)],  # ← 验证集仍用原始数据！
        eval_metric='binary_logloss',
        callbacks=[
            early_stopping(stopping_rounds=10, verbose=False),
            log_evaluation(False)
        ]
    )

    # Step 3: 预测与评估（在原始验证集上！）
    y_pred_oversample = clf_oversample.predict(X_val_final)
    y_prob_oversample = clf_oversample.predict_proba(X_val_final)[:, 1]
    recall_oversample = recall_score(y_val, y_pred_oversample)
    f1_oversample = f1_score(y_val, y_pred_oversample)
    auc_oversample = roc_auc_score(y_val, y_prob_oversample)
    print(f'[Oversampling {oversampling_ratio:.1f}] Recall: {recall_oversample:.4f}')
    print(f'[Oversampling {oversampling_ratio:.1f}] F1: {f1_oversample:.4f}')
    print(f'[Oversampling {oversampling_ratio:.1f}] AUC: {auc_oversample:.4f}')
    metrics_df_oversample = pd.DataFrame({
        'Metric': ['Recall', 'F1', 'AUC'],
        'Value': [recall_oversample, f1_oversample, auc_oversample]
    })
    metrics_df_oversample.to_csv(
        os.path.join(EXPERIMENT_DIR, 'metrics.csv'),
        index=False
    )

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob_oversample)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_oversample:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Oversampling {oversampling_ratio:.1f})')
    plt.legend()
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'roc_curve.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save Confusion Matrix
    cm_oversample = confusion_matrix(y_val, y_pred_oversample)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_oversample)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix (Oversampling {oversampling_ratio:.1f})')
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'confusion_matrix.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save model
    joblib.dump(
        clf_oversample,
        os.path.join(EXPERIMENT_DIR, 'model.pkl')
    )

    # Save oversampling 后的训练集
    pd.DataFrame(X_train_oversampled, columns=selected_features).to_csv(
        os.path.join(EXPERIMENT_DIR, 'X_train_oversampled.csv'), index=False
    )
    pd.Series(y_train_oversampled).to_csv(
        os.path.join(EXPERIMENT_DIR, 'y_train_oversampled.csv'), index=False, header=['label']
    )

# Undersampling 1.0/0.3/0.5/0.8 Experiment
under_sampling_ratios = [1.0, 0.3, 0.5, 0.8]
for undersampling_ratio in under_sampling_ratios:
    experiment_name = f'undersampling_{undersampling_ratio:.1f}'
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # Step 1: 当前数据已经是高质量候选样本
    X_train_filtered_raw = X_train_final.copy()
    y_train_filtered = y_train.copy().reset_index(drop=True)

    # Step 2: 对过滤后的数据应用特征工程
    # 直接提取全局选定的特征
    X_train_final_us = X_train_filtered_raw.reset_index(drop=True)
    y_train_final_us = y_train_filtered.reset_index(drop=True)
    
    # 类别特征编码
    cat_features_us = X_train_final_us.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if cat_features_us:
        for col in cat_features_us:
            le = LabelEncoder()
            X_train_final_us[col] = le.fit_transform(X_train_final_us[col].astype(str))
  
    print(f"[Undersampling {undersampling_ratio:.1f}] 过滤后分布: {np.bincount(y_train_final_us)}")

    # Step 3: 欠采样（使用对齐后的数据）
    minority_idx = np.where(y_train_final_us == 1)[0]
    majority_idx = np.where(y_train_final_us == 0)[0]
    n_minority = len(minority_idx)

    if undersampling_ratio > 0:
        desired_majority = int(n_minority / undersampling_ratio)
        desired_majority = min(desired_majority, len(majority_idx))
    else:
        desired_majority = len(majority_idx)

    if desired_majority < len(majority_idx):
        selected_majority = np.random.choice(majority_idx, size=desired_majority, replace=False)
        selected_indices = np.concatenate([minority_idx, selected_majority])
        X_resampled = X_train_final_us.iloc[selected_indices].reset_index(drop=True)
        y_resampled = y_train_final_us.iloc[selected_indices].reset_index(drop=True)
    else:
        X_resampled = X_train_final_us.copy()
        y_resampled = y_train_final_us.copy()

    print(f"[Undersampling {undersampling_ratio:.1f}] Undersampling 后训练集分布: {np.bincount(y_resampled)}")

    # Step 4: 训练新模型（使用平衡后的数据）
    clf_undersample = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=31,
        learning_rate=0.1,
        min_data_in_leaf=20,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    clf_undersample.fit(
        X_resampled, y_resampled,  # ← 使用 Undersampling 后的数据
        eval_set=[(X_val_final, y_val)],  # ← 验证集仍用原始数据！
        eval_metric='binary_logloss',
        callbacks=[
            early_stopping(stopping_rounds=10, verbose=False),
            log_evaluation(False)
        ]
    )

    # Step 5: 预测与评估（在原始验证集上！）
    y_pred_undersample = clf_undersample.predict(X_val_final)
    y_prob_undersample = clf_undersample.predict_proba(X_val_final)[:, 1]
    recall_undersample = recall_score(y_val, y_pred_undersample)
    f1_undersample = f1_score(y_val, y_pred_undersample)
    auc_undersample = roc_auc_score(y_val, y_prob_undersample)
    print(f'[Undersampling {undersampling_ratio:.1f}] Recall: {recall_undersample:.4f}')
    print(f'[Undersampling {undersampling_ratio:.1f}] F1: {f1_undersample:.4f}')
    print(f'[Undersampling {undersampling_ratio:.1f}] AUC: {auc_undersample:.4f}')
    metrics_df_undersample = pd.DataFrame({
        'Metric': ['Recall', 'F1', 'AUC'],
        'Value': [recall_undersample, f1_undersample, auc_undersample]
    })
    metrics_df_undersample.to_csv(
        os.path.join(EXPERIMENT_DIR, 'metrics.csv'),
        index=False
    )

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob_undersample)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_undersample:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Undersampling {undersampling_ratio:.1f})')
    plt.legend()
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'roc_curve.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save Confusion Matrix
    cm_undersample = confusion_matrix(y_val, y_pred_undersample)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_undersample)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix (Undersampling {undersampling_ratio:.1f})')
    plt.savefig(
        os.path.join(EXPERIMENT_DIR, 'confusion_matrix.png'),
        bbox_inches='tight'
    )
    plt.close()

    # Save model
    joblib.dump(
        clf_undersample,
        os.path.join(EXPERIMENT_DIR, 'model.pkl')
    )

    # Save undersampling 后的训练集
    pd.DataFrame(X_resampled, columns=selected_features).to_csv(
        os.path.join(EXPERIMENT_DIR, 'X_resampled.csv'), index=False
    )
    pd.Series(y_resampled).to_csv(
        os.path.join(EXPERIMENT_DIR, 'y_resampled.csv'), index=False, header=['label']
    )