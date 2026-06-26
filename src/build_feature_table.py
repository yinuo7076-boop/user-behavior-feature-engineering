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

cursor = conn_user.cursor()

cursor.execute("""
PRAGMA table_info(user_behavior)
""")

# 建立base table
base_table = pd.read_sql("""
SELECT DISTINCT
    user_id,
    item_id,
    item_category,
    behavior_type,
    time
FROM user_behavior
""", conn_user)
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
# 标签
base_table['label'] = (
    base_table['behavior_type'] == 4
).astype(int)
# 时间
base_table['time'] = pd.to_datetime(
    base_table['time']
)

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
feature_table.to_csv(
    'feature_table.csv',
    index=False
)
conn_user.close()
conn_item.close()
conn_category.close()
