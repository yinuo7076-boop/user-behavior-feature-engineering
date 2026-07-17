# User Behavior Feature Engineering Project

## 1. 项目简介

本项目基于用户历史行为数据，构建用户、商品和类别特征，并预测用户是否会在预测日（2014-12-18）产生购买行为。

### 项目主要包括：

- 数据清洗
- 用户/商品/类目特征工程搭建
- 数据集划分
- 特征筛选
- 模型训练与评估

### 项目亮点：

- 构建 User、Item、Category 三类特征，共58个特征
- 采用三步特征筛选方法，最终保留24个特征
- 使用 LightGBM 建立用户购买行为预测模型
- 比较 Baseline、SMOTE、Random Oversampling、Random Undersampling 四种类别不平衡处理方法

## 2. 项目结构

```text
user_behavior_feature_engineering_project/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── user_create_db.py
│   ├── item_create_db.py
│   ├── category_create_db.py
│   ├── build_feature_table.py
│   ├── split_dataset.py
│   └── feature_selection.py
│
├── docs/
│   ├── data_split_README.md
│   ├── feature_selection_README.md
│   ├── intermediate_tables_README.md
│   └── imbalanced_learning_README.md
│
├── experiments/
│   ├── baseline/
│   ├── smote_0.3/
│   ├── smote_0.5/
│   ├── smote_0.8/
│   ├── smote_1.0/
│   ├── oversampling_0.3/
│   ├── oversampling_0.5/
│   ├── oversampling_0.8/
│   ├── oversampling_1.0/
│   ├── undersampling_0.3/
│   ├── undersampling_0.5/
│   ├── undersampling_0.8/
│   └── undersampling_1.0/
│
├── data/
│   ├── middle/
│   │   ├── middle_user.db
│   │   ├── middle_item.db
│   │   └── middle_category.db
│   │
│   └── processed/
│       ├── feature_table.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
```
不同实验目录中保存对应采样策略的实验结果，包括：

```text
metrics.csv                  # 模型评估指标
model.pkl                    # 训练好的 LightGBM 模型
roc_curve.png                # ROC 曲线
confusion_matrix.png         # 混淆矩阵
X_train_smote.csv            # SMOTE 生成的训练特征（SMOTE 实验）
y_train_smote.csv
X_train_oversampled.csv      # 过采样后的训练数据（Oversampling 实验）
y_train_oversampled.csv
X_train_undersampled.csv     # 欠采样后的训练数据（Undersampling 实验）
y_train_undersampled.csv
```

## 3. 数据处理流程

### 项目整体数据处理流程如下：

```text
原始用户行为数据
        │
        ▼
构建中间特征表
(User / Item / Category)
        │
        ▼
生成候选 User-Item 样本
(2014-12-18 前历史行为)
        │
        ▼
生成Label
(2014-12-18 是否购买)
        │
        ▼
构建 Feature Table
        │
        ▼
划分 Train / Validation / Test
        │
        ▼
特征筛选
        │
        ▼
类别不平衡实验
(Baseline / SMOTE / Oversampling / Undersampling)
        │
        ▼
LightGBM 模型训练与评估
```

## 4. 特征工程

### 本项目从三个维度构建特征：

| 特征维度 | 内容 |
|---|---|
| User Feature | 用户活跃度、购买倾向、时间偏好 |
| Item Feature | 商品热度、转化率、兴趣深度 |
| Category Feature | 类别整体热度与购买趋势 |

共构建 **58 个特征** ，经过特征筛选后最终保留 **24 个特征** 用于模型训练。

### 主要特征包括：

- 浏览次数
- 加购率
- 收藏率
- 购买率
- 活跃时间
- 最近行为时间
- 用户兴趣深度
- 商品热度增长率
- 浏览未购买率
- 重复购买指数

### 详细说明见：

```text
docs/intermediate_tables_README.md
```

## 5. 数据集划分

### 本项目采用 **分层随机划分（Stratified Split）**，按照标签比例将数据划分为训练集、验证集和测试集。

| 数据集 | 占比 |
|-------|-----:|
| Train | 70% |
| Validation | 20% |
| Test | 10% |

采用`train_test_split()` 并设置`stratify=label`，确保各数据集保持一致的类别分布。

### 划分原则：

- 保持训练集、验证集和测试集中的正负样本比例一致
- 使用随机打乱后进行划分
- 保证实验可复现

### 详细说明见：

```text
docs/data_split_README.md
```

## 6. 特征筛选

### 项目采用三步特征筛选流程：

| 步骤 | 方法 |
|---|---|
| Step1 | 去除无效特征 |
| Step2 | 消除冗余 |
| Step3 | 排序 |

### 筛选原则：

- 所有筛选仅基于训练集完成，避免数据泄露
- 删除低质量与冗余特征，保留高价值特征

完成特征筛选后，项目进一步基于筛选后的特征开展类别不平衡实验，用于比较不同采样策略对模型性能的影响。

### 详细说明见：

```text
docs/feature_selection_README.md
```

## 7. 模型

### 本项目采用 **树模型 LightGBM** 作为主模型，用于：

- 特征重要性排序
- 用户购买行为预测

### 由于树模型能够较好处理：

- 非线性关系
- 特征交互
- 大规模稀疏特征

因此作为本项目重点优化模型。

## 8. 类别不平衡实验

### 本项目比较了多种类别不平衡处理方法，包括：

- Baseline
- SMOTE
- Random Oversampling
- Random Undersampling

采用 **Recall、F1-score、ROC-AUC** 作为评价指标，对不同采样策略进行了对比分析。

### 详细实验过程及结果见：

```text
docs/imbalanced_learning_README.md
```

## 9. 环境依赖

### 项目主要依赖：

```text
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.7.2
lightgbm==4.6.0
```

### 详细说明见：

```text
requirements.txt
```

## 10. 运行方法

### Step1：生成中间特征表

```bash
python src/user_create_db.py
python src/item_create_db.py
python src/category_create_db.py
```

### Step2：构建主特征表

```bash
python src/build_feature_table.py
```

### Step3：划分数据集

```bash
python src/split_dataset.py
```

### Step4：特征筛选与类别不平衡实验

```bash
python src/feature_selection.py
```

该脚本完成以下工作：

- 特征筛选
- 生成筛选后的训练集和验证集
- 基于筛选后的特征开展类别不平衡实验（Baseline、SMOTE、Random Oversampling、Random Undersampling）
- 保存不同采样策略对应的模型、评估指标及可视化结果至 `experiments/` 目录

## 11. 模型训练结果

完成特征筛选后，本项目采用 LightGBM 对筛选后的 24 个特征进行训练，并记录模型训练过程中训练集和验证集的 Binary Log Loss，用于分析模型收敛情况及泛化能力。

模型训练过程中绘制了学习曲线，结果保存在：

```text
experiments/baseline/learning_curve.png
```

从学习曲线可以观察到：

- 训练集 Binary Log Loss 随 Boosting Iterations 持续下降，说明模型能够不断学习训练数据中的特征
- 验证集 Binary Log Loss 在训练初期快速下降，随后逐渐趋于稳定
- 后期训练集与验证集损失之间存在一定差距，但验证集损失未出现明显上升，说明模型仅存在轻微过拟合，整体具有较好的泛化能力
- 本项目采用 Early Stopping 机制，在验证集性能不再提升时提前停止训练，从而避免模型进一步过拟合

训练完成后，模型保存为：

```text
experiments/baseline/model.pkl
```

模型文件大小约为 **294 KB**，便于后续加载、部署及预测使用。
训练完成后，各实验均自动保存模型评估指标、ROC Curve、Confusion Matrix 及训练好的模型文件，便于不同类别平衡策略之间的性能比较与实验复现。

在模型评估阶段，本项目进一步比较了 Baseline、SMOTE、Random Oversampling 和 Random Undersampling 四种类别不平衡处理策略，并保存了各实验对应的模型、评估指标及可视化结果（ROC Curve、Confusion Matrix 等）。

总体来看，LightGBM 能够较好地学习用户行为特征，不同采样策略对模型的 Recall、F1-score 和 ROC-AUC 均产生了一定影响。其中，Random Oversampling 在模型综合性能方面表现较优，而 Random Undersampling 能够进一步提高 Recall，但会带来更多误分类，需要结合具体应用场景进行权衡。

关于类别不平衡实验的实验设计、评价指标、模型性能对比及结果分析，请参考：

```text
docs/imbalanced_learning_README.md
```

## 12. 数据集

由于数据文件较大（超过 GitHub 文件大小限制），本项目未上传数据文件。

### 12.1 原始数据（Raw Data）

项目运行依赖原始数据文件：

```text
raw_data/
└── data_min.csv
```

该文件未包含在本项目中，请下载后放置到本地任意目录，并根据实际存放位置修改代码中的数据读取路径（本项目开发环境中使用 raw_data/data_min.csv 作为原始数据路径）。

### 12.2 项目数据（Project Data）

项目运行过程中生成的中间数据及处理结果位于：

```text
data/
├── middle/
│   ├── middle_user.db
│   ├── middle_item.db
│   ├── middle_category.db
│
└── processed/
    ├── feature_table.csv
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── X_train_selected.csv
    └── X_val_selected.csv
```

由于上述数据文件体积较大，因此未上传至 GitHub。

可通过以下链接下载：

> **UserBehaviorDataset**  
> 链接：https://pan.baidu.com/s/1QQXo60MTZ1H7Usfo6YIZqA  
> 提取码：6aeg

下载内容包括：

- raw_data/data_min.csv（原始数据）
- data/（项目运行过程中生成的数据）

下载完成后，请保持上述目录结构不变，并放置到项目对应位置后再运行代码。