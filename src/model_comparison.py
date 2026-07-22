import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation,
)
from xgboost import XGBClassifier

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

# Model Comparison
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
MODEL_DIR = os.path.join(EXPERIMENTS_DIR, "model_comparison")
os.makedirs(MODEL_DIR, exist_ok=True)
models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    ),
    "LightGBM": LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=31,
        learning_rate=0.1,
        min_data_in_leaf=20,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
}
comparison_results = []
roc_data = []
for model_name, model in models.items():
    print("=" * 50)
    print(f"Training {model_name}")
    print("=" * 50)
    # Train
    if model_name == "LightGBM":
        model.fit(
            X_train_final,
            y_train,
            eval_set=[(X_val_final, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(stopping_rounds=10, verbose=False),
                log_evaluation(False)
            ]
        )
    else:
        model.fit(
            X_train_final,
            y_train
        )
    # Save model
    filename = model_name.lower().replace(" ", "_")
    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            f"{filename}.pkl"
        )
    )
    
    # Predict
    y_pred = model.predict(X_val_final)
    y_prob = model.predict_proba(X_val_final)[:, 1]
    precision = precision_score(
        y_val,
        y_pred
    )
    recall = recall_score(
        y_val,
        y_pred
    )
    f1 = f1_score(
        y_val,
        y_pred
    )
    auc = roc_auc_score(
        y_val,
        y_prob
    )
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_data.append({
        "name": model_name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    })
    comparison_results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    })
    print(f"{model_name}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

# Save comparison results
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(
    os.path.join(MODEL_DIR, "model_comparison_metrics.csv"),
    index=False
)
print("Model Comparison Results: ")
print(comparison_df)

# ROC Comparison
plt.figure(figsize=(7, 6))
for roc in roc_data:
    plt.plot(
        roc["fpr"],
        roc["tpr"],
        linewidth=2,
        label=f'{roc["name"]} (AUC={roc["auc"]:.4f})'
    )
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    linewidth=1
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    os.path.join(
        MODEL_DIR,
        "roc_curve_comparison.png"
    ),
    dpi=300
)
plt.close()