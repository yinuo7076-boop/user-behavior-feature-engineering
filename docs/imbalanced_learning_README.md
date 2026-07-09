# 类别不平衡实验说明（Imbalanced Learning）

本项目预测用户是否会在预测日（2014-12-18）产生购买行为。

训练集中购买样本数量明显少于未购买样本，属于典型的类别不平衡二分类问题。

训练集类别分布如下：

| 类别 | 样本数 | 占比 |
|------|-------:|------:|
| Negative（未购买） | 7146 | 90.90% |
| Positive（购买） | 715 | 9.10% |

类别不平衡可能导致模型偏向预测多数类，从而降低对购买行为的识别能力。

因此，本项目进一步比较不同类别不平衡处理方法对模型性能的影响。

## 1. 实验设计

本项目比较以下四种类别不平衡处理方法：

- Baseline（原始训练集）
- SMOTE
- Random Oversampling
- Random Undersampling

所有实验均基于：

- 相同的数据集划分
- 相同的特征集合
- 相同的 LightGBM 模型参数

仅改变训练集的采样策略，以保证实验结果具有可比性。

---

## 2. 数据处理原则

### 2.1 负样本定义

原始负样本包括所有未发生购买行为的用户-商品交互。

对于 Random Undersampling 实验，为了提高负样本质量，首先筛选出具有较高购买意图但最终未购买的样本，仅保留：

- 加购（behavior_type=2）
- 收藏（behavior_type=3）

对应的未购买交互。

随后在此基础上进行随机欠采样。

---

### 2.2 数据集划分

所有实验均使用相同的数据集划分方式：

- Train：70%
- Validation：20%
- Test：10%

采用 `train_test_split()` 并设置 `stratify=label`，保证三个数据集具有相近的类别分布。

所有采样操作仅作用于训练集，验证集和测试集保持原始数据分布。

---

## 3. 实验设计

### 3.1 Baseline

直接使用原始训练集训练LightGBM模型，不进行任何采样处理。

作为所有实验的性能基准。

---

### 3.2 SMOTE

采用 SMOTE（Synthetic Minority Over-sampling Technique）生成新的少数类样本。

实验分别测试以下采样比例：

- 0.3
- 0.5
- 0.8
- 1.0

所有生成样本仅用于训练集。

---

### 3.3 Random Oversampling

采用随机重复采样方式扩充正样本。

实验分别测试以下采样比例：

- 0.3
- 0.5
- 0.8
- 1.0

验证集保持不变。

---

### 3.4 Random Undersampling

步骤：

1. 保留加购或收藏但未购买的负样本
2. 对筛选后的负样本进行随机欠采样

提高了训练数据中负样本的质量，从而增强模型对购买行为的学习能力。

实验分别测试以下采样比例：

- 0.3
- 0.5
- 0.8
- 1.0

---

## 4. 模型

所有实验均采用相同的LightGBM参数进行训练，以保证实验之间具有可比性。

主要参数如下：

```python
objective='binary'
metric='binary_logloss'
num_leaves=31
learning_rate=0.1
n_estimators=100
random_state=42
```

训练过程中采用 Early Stopping：

```python
early_stopping(stopping_rounds=10)
```

避免模型过拟合。

---

## 5. 评价指标

所有实验均在相同的验证集上进行评估。

采用以下指标：

| 指标 | 说明 |
|------|------|
| Recall | 正样本召回率 |
| F1-score | Precision 与 Recall 的综合指标 |
| ROC-AUC | 模型整体分类能力 |

同时保存：

- ROC Curve
- Confusion Matrix
- Model
- Metrics

用于不同实验之间的性能比较。

---

## 6. 数据泄露控制

为了保证实验结果可靠，本项目采取以下措施：

- 所有采样操作仅应用于训练集
- 验证集和测试集保持原始数据分布
- 特征筛选仅基于训练集完成
- 所有实验使用相同的特征集合和模型参数

---

## 7. 实验结果路径

实验结果保存在不同实验目录中：

```text
experiments/
├── baseline/
├── smote_0.3/
├── smote_0.5/
├── smote_0.8/
├── smote_1.0/
├── oversampling_0.3/
├── oversampling_0.5/
├── oversampling_0.8/
├── oversampling_1.0/
├── undersampling_0.3/
├── undersampling_0.5/
├── undersampling_0.8/
└── undersampling_1.0/
```

每个实验目录均包含：

- metrics.csv
- model.pkl
- roc_curve.png
- confusion_matrix.png

以及对应实验生成的数据文件。

---

## 8. 实验结果对比

各类别不平衡处理方法在验证集上的综合性能最佳实验结果如下：

| 方法 | Recall | F1-score | ROC-AUC |
|------|--------:|---------:|---------:|
| Baseline | 0.4902 | 0.5917 | 0.9286 |
| SMOTE (sampling ratio = 1.0) | 0.5833 | 0.5819 | 0.9102 |
| Random Oversampling (sampling ratio = 1.0) | 0.7549 | 0.6273 | 0.9250 |
| Random Undersampling (sampling ratio = 0.3) | 0.6912 | 0.5767 | 0.9219 |

完整实验结果可查看各实验目录下的 `metrics.csv`、`roc_curve.png` 和 `confusion_matrix.png`。

从实验结果可以看出：

- 与 Baseline 相比，三种采样方法均提高了模型对购买样本的识别能力（Recall）。
- Oversampling 在 Recall、F1-score 和 ROC-AUC 三项指标之间取得了较好的平衡，综合性能最佳。
- SMOTE 能够提升 Recall，但对 F1-score 和 ROC-AUC 的改善相对有限。
- Undersampling 能够进一步提高 Recall，但随着欠采样比例增大，F1-score 会有所下降，因此需要在召回率和整体性能之间进行权衡。

---

## 9. 实验总结

本项目系统比较了 Baseline、SMOTE、Random Oversampling 和 Random Undersampling 四种类别不平衡处理方法。

实验结果表明，不同类别不平衡处理策略在 Recall、F1-score 和 ROC-AUC 等指标上的表现存在差异。

通过对多种采样策略进行对比分析，为后续模型优化和类别不平衡处理方法的选择提供了参考。

> 本实验建立在完整的特征工程、数据集划分和特征筛选流程之上，详细过程可参考：
>
> - `docs/intermediate_tables_README.md`
> - `docs/data_split_README.md`
> - `docs/feature_selection_README.md`