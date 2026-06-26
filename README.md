# User Behavior Feature Engineering Project

## 1. 项目简介

本项目基于用户行为数据，完成用户购买行为预测任务。

项目主要包括：

- 数据清洗；
- 用户/商品/类目特征工程搭建
- 数据集划分
- 特征筛选
- 模型训练与评估

项目目标是通过用户历史行为数据，构建行为特征，并预测用户是否会产生购买行为

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

项目整体数据处理流程如下：

user_behavior.csv  
->  
用户/商品/类目特征构建:  
middle_user.db / middle_item.db / middle_category.db  
->  
特征表拼接  
->  
feature_table.csv  
->  
时间窗口划分  
->  
train / validation / test  
->  
特征筛选  
->  
模型训练与评估

## 4. 特征工程

本项目从三个维度构建特征：

| 特征维度 | 内容 |
|---|---|
| User Feature | 用户活跃度、购买倾向、时间偏好 |
| Item Feature | 商品热度、转化率、兴趣深度 |
| Category Feature | 类别整体热度与购买趋势 |

主要特征包括：

- 浏览次数
- 加购率
- 收藏率
- 购买率
- 活跃时间
- 最近行为时间
- 用户兴趣深度
- 商品热度增长率
- 浏览未购买率等

详细说明见：

```text
docs/intermediate_tables.md
```

## 5. 数据集划分

采用时间窗口划分方式：

| 数据集 | 时间范围 |
|---|---|
| Train | 2014-11-18 ~ 2014-12-10 |
| Validation | 2014-12-10 ~ 2014-12-15 |
| Test | 2014-12-15 ~ 2014-12-18 |

划分原则：

- 避免数据泄露
- 保持时间顺序
- 保证label分布相近
- 保证工作日与全时间段覆盖

详细说明见：

```text
docs/data_split_README.md
```

## 6. 特征筛选

项目采用三步特征筛选流程：

| 步骤 | 方法 |
|---|---|
| Step1 | 去除无效特征 |
| Step2 | 消除冗余 |
| Step3 | 排序 |

筛选原则：

- 所有筛选仅基于训练集完成，避免数据泄露
- 删除低质量与冗余特征，保留高价值特征

详细说明见：

```text
docs/feature_selection_README.md
```

## 7. 模型

本项目采用树模型 LightGBM 作为主模型，用于：

- 特征重要性排序
- 用户购买行为预测

由于树模型能够较好处理：

- 非线性关系
- 特征交互
- 大规模稀疏特征

因此作为本项目重点优化模型。

同时使用 Logistic Regression 模型，用于简单性能对照。

## 8. 环境依赖

项目主要依赖：

```text
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.7.2
lightgbm==4.6.0
```

详细说明见：

```text
requirements.txt
```

## 9. 运行方法

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

## 10. 项目说明

本项目重点关注：

- 特征工程
- 时间窗口划分
- 数据泄露控制
- 特征筛选流程
- 用户行为建模

适用于：

- 推荐系统
- 用户行为分析
- 电商数据挖掘等场景

### 11. Dataset

由于数据文件较大（超过 GitHub 限制），本项目未上传 data 文件夹。

数据可以通过以下链接下载：

https://pan.baidu.com/s/1WcanaXngjIcaUTAvlYjubA 提取码: uub5

下载后请解压，并放在项目根目录，结构如下：

data/