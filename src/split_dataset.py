import os
import pandas as pd
from sklearn.model_selection import train_test_split
# 获取当前脚本所在目录（src/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 processed 数据目录路径（相对于项目根目录）
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
# 读取 feature_table.csv
feature_table = pd.read_csv(os.path.join(processed_dir, 'feature_table.csv'))

train_df, temp_df = train_test_split(
    feature_table,
    test_size=0.3,
    random_state=42,
    stratify=feature_table['label']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,
    random_state=42,
    stratify=temp_df['label']
)

# 打印数据大小
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# label分布
print("\nTrain label distribution:")
print(train_df['label'].value_counts())

print("\nValidation label distribution:")
print(val_df['label'].value_counts())

print("\nTest label distribution:")
print(test_df['label'].value_counts())

# 保存
train_df.to_csv(
    os.path.join(processed_dir, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(processed_dir, "val.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(processed_dir, "test.csv"),
    index=False
)