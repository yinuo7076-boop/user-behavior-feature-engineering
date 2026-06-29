import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss  
from itertools import combinations
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# 读取数据
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')
# 删除不作为特征的列
drop_cols = [
    'user_id',
    'item_id',
    'time',
    'behavior_type',
    'label'
]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['label']

X_val = val_df.drop(columns=drop_cols)
y_val = val_df['label']

X_test = test_df.drop(columns=drop_cols)
# 仅当 test.csv 包含 label 时才提取，否则设为 None
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
print(f"原始特征数: {X_train.shape[1]}")
print(f"步骤1后: {X_train_step1.shape[1]}")
print(f"步骤2后: {X_train_step2.shape[1]}")
print(f"最终保留: {len(selected_features)}")

X_train_final.to_csv('X_train_selected.csv', index=False)
X_val_final.to_csv('X_val_selected.csv', index=False)
pd.Series(selected_features).to_csv('selected_features.txt', index=False, header=False)