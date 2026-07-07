import os
import sqlite3
import pandas as pd
# 获取当前脚本所在目录（即 src/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在目录（即 src/）
middle_dir = os.path.join(script_dir, '..', 'data', 'middle')
# 连接数据库（使用绝对路径）
conn_user = sqlite3.connect(os.path.join(middle_dir, 'middle_user.db'))
conn_item = sqlite3.connect(os.path.join(middle_dir, 'middle_item.db'))
conn_category = sqlite3.connect(os.path.join(middle_dir, 'middle_category.db'))
# 读取3个特征表
user_feature = pd.read_sql(
    "SELECT * FROM user_feature",
    conn_user
)
item_feature = pd.read_sql(
    "SELECT * FROM item_feature",
    conn_item
)
category_feature = pd.read_sql(
    "SELECT * FROM category_feature",
    conn_category
)

# 所有历史行为（12月18日前）
history = pd.read_sql("""
SELECT
    user_id,
    item_id,
    item_category,
    behavior_type,
    time
FROM user_behavior
WHERE time < '2014-12-18'
""", conn_user)

history['time'] = pd.to_datetime(history['time'])
# 只保留浏览、收藏、加购行为作为候选样本
candidate = history[
    history['behavior_type'].isin([1, 2, 3])
]
base_table = candidate[
    ['user_id', 'item_id', 'item_category']
].drop_duplicates()

# 统一格式
user_feature['user_id'] = user_feature['user_id'].astype(str)
item_feature['item_id'] = item_feature['item_id'].astype(str)
category_feature['item_category'] = (
    category_feature['item_category']
    .astype(str)
)
base_table['user_id'] = base_table['user_id'].astype(str)
base_table['item_id'] = base_table['item_id'].astype(str)
base_table['item_category'] = (
    base_table['item_category']
    .astype(str)
)

# 预测日（12月18日）的购买记录
label_df = pd.read_sql("""
SELECT DISTINCT
    user_id,
    item_id
FROM user_behavior
WHERE behavior_type = 4
AND time >= '2014-12-18'
AND time < '2014-12-19'
""", conn_user)

# 类型保持一致
label_df['user_id'] = label_df['user_id'].astype(str)
label_df['item_id'] = label_df['item_id'].astype(str)

# 给购买样本打标签
label_df['label'] = 1

# 合并标签
base_table = base_table.merge(
    label_df,
    on=['user_id', 'item_id'],
    how='left'
)

# 未购买填0
base_table['label'] = (
    base_table['label']
    .fillna(0)
    .astype(int)
)

# 正样本
positive = base_table[base_table['label'] == 1]
# 负样本
negative = base_table[base_table['label'] == 0]
# 随机抽取10倍负样本
negative = negative.sample(
    n=len(positive) * 10,
    random_state=42
)
# 合并
base_table = pd.concat(
    [positive, negative],
    ignore_index=True
)
# 打乱
base_table = base_table.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)
print("Sampled label distribution:")
print(base_table['label'].value_counts())

# 连接user特征
feature_table = base_table.merge(
    user_feature,
    on='user_id',
    how='left'
)
# 连接item特征
feature_table = feature_table.merge(
    item_feature,
    on='item_id',
    how='left'
)
# 链接category特征
feature_table = feature_table.merge(
    category_feature,
    on='item_category',
    how='left'
)
# 缺失值处理
feature_table = feature_table.fillna(0)

# 保存主特征表
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')

os.makedirs(processed_dir, exist_ok=True)

feature_table.to_csv(
    os.path.join(processed_dir, 'feature_table.csv'),
    index=False
)
print("Feature table shape:", feature_table.shape)

conn_user.close()
conn_item.close()
conn_category.close()