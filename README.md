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

- 构建User、Item、Category三类特征，共58个特征
- 采用三步特征筛选流程，最终保留24个特征
- 使用LightGBM建立购买预测模型
- 比较Baseline、SMOTE、Random Oversampling、Random Undersampling四种类别不平衡处理方法

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
│   └── intermediate_tables.md
│   └── imbalanced_learning_README.md
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
docs/intermediate_tables.md
```

## 5. 数据集划分

### 本项目采用 **分层随机划分（Stratified Split）**，按照标签比例将数据划分为训练集、验证集和测试集。

| 数据集 | 占比 |
|-------|-----:|
| Train | 70% |
| Validation | 20% |
| Test | 10% |

采用```train_test_split()``` 并设置```stratify=label```，确保各数据集保持一致的类别分布。

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

### Step4：特征筛选

```bash
python src/feature_selection.py
```

## 11. 数据集

由于Data文件较大（超过 GitHub 限制），本项目未上传 Data 文件夹。

### 数据可以通过以下链接下载：

https://pan.baidu.com/s/1WcanaXngjIcaUTAvlYjubA 提取码: uub5

### 下载后请解压，并放在项目根目录，结构如下：

data/