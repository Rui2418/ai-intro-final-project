分类模型相关命令：

1. 训练 baseline：
python -m src.train_baseline

2. 对比传统分类模型：
python -m src.experiments.train_compare

3. 联合搜索最佳传统模型：
python -m src.experiments.joint_search

4. 训练最终分类模型：
python -m src.train_final_model

运行最终训练脚本后会生成：
- models/final_classifier_pipeline.joblib
- outputs/val_predictions_final_model.csv
- outputs/final_model_metrics.txt



# 工作总结：分类模型实验、调优与分析

本阶段主要负责“可解释谣言检测”项目中的**分类模型部分**，包括数据读取、文本预处理、baseline 复现、模型对比、参数调优、联合搜索、交叉验证、错误分析、深度学习模型尝试，以及最终分类模型训练脚本整理。

---

## 1. 数据读取与预处理理解

首先确认数据集结构：

```text
train.csv：训练集
val.csv：验证集
```

数据字段包括：

```text
id
text
label
event
```

其中：

```text
text：输入文本
label：分类标签，0 表示非谣言，1 表示谣言
event：事件编号
```

项目已有统一文本预处理函数 `clean_text()`，主要完成：

```text
HTML 反转义
文本小写化
URL 替换为 <url>
@用户 替换为 <user>
合并多余空格
```

该预处理策略后续在实验中被验证为当前效果最优。

---

## 2. baseline 复现

复现了原始 baseline：

```text
TF-IDF + Logistic Regression
```

验证集结果：

```text
Accuracy = 0.8429
```

该模型作为后续模型对比和优化的基准。

---

## 3. 传统机器学习模型对比

新增模型对比实验，比较：

```text
TF-IDF + Logistic Regression
TF-IDF + Linear SVM
TF-IDF + Naive Bayes
```

初步模型对比结果：

| 模型                           | Accuracy | Precision | Recall |     F1 |
| ---------------------------- | -------: | --------: | -----: | -----: |
| TF-IDF + Logistic Regression |   0.8429 |    0.8544 | 0.7714 | 0.8108 |
| TF-IDF + Linear SVM          |   0.8554 |    0.8634 | 0.7943 | 0.8274 |
| TF-IDF + Naive Bayes         |   0.8628 |    0.8797 | 0.7943 | 0.8348 |

初步结论：

```text
在相同 TF-IDF 设置下，Naive Bayes 表现最好。
```

---

## 4. TF-IDF 参数调优

在 Naive Bayes 表现较好的基础上，进一步对 TF-IDF 和 Naive Bayes 参数进行调优，搜索参数包括：

```text
ngram_range
min_df
max_df
sublinear_tf
alpha
```

调优后的最优 Naive Bayes 组合为：

```text
ngram_range = (1, 2)
min_df = 2
max_df = 0.9
sublinear_tf = False
alpha = 1.0
```

结果：

```text
Accuracy = 0.8653
Precision = 0.8805
Recall = 0.8000
F1 = 0.8383
```

相比初始 Naive Bayes：

```text
Accuracy: 0.8628 → 0.8653
```

有小幅提升。

---

## 5. 文本预处理策略对比

考虑到文本预处理会影响 TF-IDF 特征，进一步对比多种预处理策略：

```text
original clean_text
remove_url_user
hashtag_split
remove_punctuation
```

结果显示：

```text
original clean_text 表现最好
Accuracy = 0.8653
```

因此最终没有直接修改公共预处理文件 `src/preprocess.py`，继续保留原始预处理策略。

这一实验说明：在当前数据集上，保留 `<url>` 和 `<user>` 等社交媒体结构标记是有价值的，过度删除信息反而可能损失特征。

---

## 6. 联合搜索：预处理 + TF-IDF + 分类器

意识到分阶段调参可能只得到局部最优，因此进一步进行了更完整的联合搜索，同时搜索：

```text
预处理策略
TF-IDF 参数
分类器类型
分类器参数
```

联合搜索比较了：

```text
Logistic Regression
Linear SVM
Naive Bayes
```

最终最优组合为：

```text
preprocess strategy: original
ngram_range = (1, 3)
min_df = 1
max_df = 0.9
sublinear_tf = True
classifier = Linear SVM
C = 2.0
class_weight = balanced
```

验证集结果：

```text
Accuracy = 0.8728
Precision = 0.8924
Recall = 0.8057
F1 = 0.8468
```

相比前一阶段 tuned Naive Bayes：

```text
Accuracy: 0.8653 → 0.8728
```

进一步提升了约 0.75 个百分点。

该结果说明：
**固定 Naive Bayes 后得到的局部最优，并不一定是整体最优。分类器类型和 TF-IDF 参数之间存在交互影响。**

---

## 7. 交叉验证稳定性分析

考虑到多轮调参和联合搜索都使用了 `val.csv`，可能存在一定验证集适配风险，因此对当前最优模型进行了 5 折分层交叉验证。

交叉验证对象：

```text
original preprocess + TF-IDF + Linear SVM
```

交叉验证结果：

```text
Accuracy mean = 0.8514
Accuracy std = 0.0194

Precision mean = 0.8408
Precision std = 0.0349

Recall mean = 0.8153
Recall std = 0.0186

F1 mean = 0.8275
F1 std = 0.0203
```

结论：

```text
模型在不同训练集划分下整体表现较稳定，但 val.csv 上的 0.8728 可能略偏乐观。
```

因此报告中应严谨表述为：

```text
该模型在 val.csv 上取得最高结果，但由于多轮模型选择基于 val.csv，仍可能存在一定验证集适配风险。
```

---

## 8. 错误分析

基于当前最优模型：

```text
original preprocess + TF-IDF + Linear SVM
```

对 `val.csv` 预测错误样本进行分析。

整体结果：

```text
Total validation samples = 401
Correct predictions = 350
Error predictions = 51
Accuracy = 0.8728
```

混淆矩阵：

```text
true 0 predicted 0: 209
true 0 predicted 1: 17
true 1 predicted 0: 34
true 1 predicted 1: 141
```

错误类型：

```text
False Positive：真实非谣言被误判为谣言 = 17
False Negative：真实谣言被误判为非谣言 = 34
```

结论：

```text
模型更容易出现谣言漏判，即将真实谣言误判为非谣言。
```

按 event 统计发现，不同事件上的准确率存在差异：

```text
event 0 accuracy = 0.7692，但样本数仅 13，需要谨慎解释
event 4 accuracy = 0.8478，低于整体准确率
event 1 accuracy = 0.8624
event 5 accuracy = 0.8678
event 6 accuracy = 0.8876
```

该部分可用于报告中的错误案例分析和泛化能力讨论。

---

## 9. DistilBERT 深度学习模型尝试

在传统机器学习模型基础上，进一步尝试了深度学习模型：

```text
distilbert-base-uncased
```

使用 GPU 训练，环境确认：

```text
PyTorch = 2.6.0+cu124
CUDA available = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

初始 3 epoch 结果：

```text
Accuracy = 0.8354
F1 = 0.8103
```

将 epoch 调整到 5 后，结果提升为：

```text
Accuracy = 0.8653
Precision = 0.8457
Recall = 0.8457
F1 = 0.8457
```

结论：

```text
DistilBERT 微调后效果明显提升，接近 tuned Naive Bayes，但仍略低于联合搜索得到的 TF-IDF + Linear SVM 最优结果。
```

因此最终不采用 DistilBERT 作为主分类模型，而是作为深度学习扩展实验写入报告。

---

## 10. 最终分类模型确定

综合所有实验，最终建议采用：

```text
original preprocess + TF-IDF + Linear SVM
```

最终参数：

```text
TF-IDF:
ngram_range = (1, 3)
min_df = 1
max_df = 0.9
sublinear_tf = True

Linear SVM:
C = 2.0
class_weight = balanced
random_state = 42
```

最终模型在 `val.csv` 上：

```text
Accuracy = 0.8728
Precision = 0.8924
Recall = 0.8057
F1 = 0.8468
```

---

## 11. 新增最终模型训练脚本

新增：

```text
src/train_final_model.py
```

作用：

```text
使用已确定的最佳参数训练最终分类模型
保存完整 sklearn Pipeline
输出验证集预测结果和指标
```

运行命令：

```bash
python -m src.train_final_model
```

生成文件：

```text
models/final_classifier_pipeline.joblib
outputs/val_predictions_final_model.csv
outputs/final_model_metrics.txt
```

因为 `.gitignore` 忽略 `models/` 和 `outputs/`，仓库主要提交脚本，运行后可复现生成模型和结果。

---

## 12. 实验脚本结构整理

为避免 `src/` 目录混乱，将实验相关脚本整理到：

```text
src/experiments/
```

推荐结构：

```text
src/
├── config.py
├── preprocess.py
├── predict.py
├── explain.py
├── train_baseline.py
├── train_final_model.py
└── experiments/
    ├── train_compare.py
    ├── tune_tfidf.py
    ├── tune_preprocess.py
    ├── joint_search.py
    ├── cross_validate_best_model.py
    ├── analyze_errors.py
    └── train_distilbert.py
```

其中：

```text
src/：核心运行代码
src/experiments/：实验、调参、分析脚本
```

---

## 13. 与成员 3 的衔接

分类模块最终提供：

```text
输入 text
输出 label
```

解释模块可以基于：

```text
text + label
```

生成自然语言解释 `reason`。

最终系统接口建议为：

```python
{
    "label": 1,
    "reason": "..."
}
```

最终模型可以通过：

```python
import joblib
from src.preprocess import clean_text

model = joblib.load("models/final_classifier_pipeline.joblib")

text = "example tweet text"
label = model.predict([clean_text(text)])[0]
```

成员 3 如果需要模型高权重词增强解释，可以基于最终 `TF-IDF + Linear SVM` 模型提取：

```python
pipeline.named_steps["tfidf"].get_feature_names_out()
pipeline.named_steps["classifier"].coef_
```

其中：

```text
正权重较大的词/短语更支持 label=1，即谣言
负权重较大的词/短语更支持 label=0，即非谣言
```

---

# 最终成果概括

本次工作完成了成员 2 负责的分类模型模块，包括：

```text
1. 理解数据结构与任务目标
2. 复现 baseline 模型
3. 对比 Logistic Regression、SVM、Naive Bayes
4. 对 TF-IDF 和 Naive Bayes 进行参数调优
5. 对文本预处理策略进行实验对比
6. 进行预处理、TF-IDF、分类器的联合搜索
7. 得到当前最优传统模型 TF-IDF + Linear SVM
8. 进行 5 折交叉验证，评估模型稳定性
9. 进行错误分析，统计误报、漏报和 event 维度表现
10. 尝试 DistilBERT 深度学习模型
11. 整理实验脚本结构
12. 新增最终分类模型训练脚本
13. 为成员 3 的解释模块提供模型接口和衔接方式
```

最终建议采用：

```text
original preprocess + TF-IDF + Linear SVM
```

作为项目最终分类模型。
