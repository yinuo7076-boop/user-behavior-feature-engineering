import os
import pandas as pd
# 获取当前脚本所在目录（src/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 processed 数据目录路径（相对于项目根目录）
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
# 读取 feature_table.csv
feature_table = pd.read_csv(os.path.join(processed_dir, 'feature_table.csv'))
# 时间格式转换
feature_table['time'] = pd.to_datetime(
    feature_table['time']
)
# 时间范围检查
feature_table = feature_table.sort_values(
    by='time'
)
# 计算切分点
n = len(feature_table)
train_end = int(n * 0.7)
val_end = int(n * 0.9)
# 按时间序列切分
train_df = feature_table.iloc[:train_end]
val_df = feature_table.iloc[
    train_end:val_end
]
test_df = feature_table.iloc[val_end:]
# 检查
print("Train:")
print(train_df['time'].min())
print(train_df['time'].max())

print("Validation:")
print(val_df['time'].min())
print(val_df['time'].max())

print("Test:")
print(test_df['time'].min())
print(test_df['time'].max())
# label分布
print(train_df['label'].mean())
print(val_df['label'].mean())
print(test_df['label'].mean())
# 每天覆盖
print(train_df['time'].dt.date.value_counts())
# 每小时覆盖
print(train_df['time'].dt.hour.value_counts())
# 保存
train_df.to_csv(
    'train.csv',
    index=False
)

val_df.to_csv(
    'val.csv',
    index=False
)

test_df.to_csv(
    'test.csv',
    index=False
)