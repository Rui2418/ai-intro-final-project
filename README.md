# ai-intro-final-project

人工智能导论大作业项目仓库，题目为**可解释的谣言检测**。

## 项目目标

构建一个能够对输入文本进行谣言分类并输出判断依据的系统。

系统输入一段英文社交媒体文本，输出：

- `label`：分类结果，`0` 表示非谣言，`1` 表示谣言
- `reason`：一段自然语言说明判断依据

## 当前数据

仓库当前已经包含课程数据集：

```text
rumer2026/
├─ train.csv
└─ val.csv
```

已确认字段包括：

- `id`
- `text`
- `label`
- `event`

## 项目结构

```text
ai-intro-final-project/
├─ rumer2026/                # 原始数据集
├─ src/                      # 主要代码
│  ├─ config.py              # 路径与字段配置
│  ├─ preprocess.py          # 文本清洗
│  ├─ train_baseline.py      # baseline 训练与验证
│  ├─ train_bert.py          # BERT 分类器训练与验证
│  ├─ optimize_transformers.py # CUDA Transformer 调参实验
│  ├─ build_best_cuda_ensemble.py # 最佳集成结果复现
│  ├─ predict.py             # baseline / BERT 预测
│  └─ explain.py             # 判断依据生成
├─ models/                   # 训练得到的模型文件
├─ outputs/                  # 预测结果、实验指标、解释样例
├─ notebooks/                # 数据分析与实验草稿
├─ report/                   # 报告和插图
├─ README.md
├─ requirements.txt
├─ main.py                   # 单条文本预测入口
└─ 小组分工方案.md
```

## 环境安装

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 方法路线

本项目采用由浅入深的建模路线：

1. 首先构建 `TF-IDF + Logistic Regression` baseline，用于验证数据读取、预处理、训练和预测流程，并作为后续模型的性能对照。
2. 随后改用 BERT/RoBERTa 风格的预训练 Transformer 模型进行微调，利用上下文语义表示提升谣言检测性能。
3. 最后结合传统模型和多个 RoBERTa 变体在验证集上的预测结果，构建多数投票集成，进一步提高验证集准确率和稳定性。

因此，baseline 是对照实验和流程验证的一部分；最终提交的最高验证集结果来自 BERT/RoBERTa 系列模型与传统模型预测的集成方案。

## 训练 baseline

```bash
python -m src.train_baseline
```

运行后将：

- 读取 `rumer2026/train.csv` 和 `rumer2026/val.csv`
- 训练 `TF-IDF + Logistic Regression` baseline
- 在验证集上输出 Accuracy 和分类报告
- 将模型保存到 `models/baseline_pipeline.joblib`
- 将验证集预测结果保存到 `outputs/val_predictions.csv`
- 将前 20 条解释样例保存到 `outputs/examples.csv`
- 将指标保存到 `outputs/metrics.txt`

## 训练 BERT 分类器

```bash
python -m src.train_bert
```

可选参数示例：

```bash
python -m src.train_bert --model-name distilbert-base-uncased --epochs 3 --batch-size 8 --max-length 256
```

如果机器不能访问 Hugging Face，可以把 `--model-name` 指向一个已经下载好的本地 Hugging Face 模型目录；该目录需要包含 `config.json`、tokenizer 文件和模型权重文件。

运行后将：

- 下载并微调一个英文预训练 Transformer 分类模型
- 自动在有 CUDA 时使用显卡训练，没有则回退到 CPU
- 将最佳模型保存到 `models/bert_classifier/`
- 将验证集预测结果保存到 `outputs/bert_val_predictions.csv`
- 将指标保存到 `outputs/bert_metrics.txt`

## 最终集成验证集结果

当前保留的最佳可复现方案是三模型多数投票集成，验证集结果为：

```text
Validation accuracy: 0.9027
Validation F1: 0.8883
```

集成成员为：

- `outputs/val_predictions_best_joint_all_models.csv`
- `outputs/continue_best_lr5e6_ep2_val_predictions.csv`
- `outputs/roberta_lr2e5_len256_ep4_seed123_val_predictions.csv`
- `outputs/roberta_lr2e5_len256_ep4_seed2024_val_predictions.csv`

脚本会先构建前三个成员的基础集成，再与 `val_predictions_best_joint_all_models` 和 `roberta_lr2e5_len256_ep4_seed2024` 做最终多数投票。

复现最终指标和预测文件：

```bash
python -m src.build_best_cuda_ensemble
```

运行后会生成：

- `outputs/best_cuda_ensemble_metrics.txt`
- `outputs/best_cuda_ensemble_val_predictions.csv`

注意：脚本需要在项目根目录用 `python -m ...` 模块方式运行，不要直接用文件路径执行。

## 单条文本预测

训练完成后运行：

```bash
python main.py --text "Breaking news example text"
```

如果要显式指定模型：

```bash
python main.py --text "Breaking news example text" --model-type bert
python main.py --text "Breaking news example text" --model-type baseline
```

输出格式示例：

```text
Label: 1
Reason: The text is classified as rumor because it contains rumor-related cues such as breaking; and it does not provide a clearly verifiable source in the text.
```

## 批量预测

训练完成后，也可以对 CSV 文件批量预测。输入文件需要包含 `text` 字段。

```bash
python -m src.predict --input rumer2026/val.csv --output outputs/batch_predictions.csv --model-type bert
```

## 当前版本说明

当前仓库已经同时支持：

- `TF-IDF + Logistic Regression` baseline
- `BERT / RoBERTa` 风格预训练文本分类模型微调
- 三模型/两层多数投票集成，当前验证集最佳准确率为 `0.9027`
- 基于规则的解释模块输出
- 模型对比、调参、交叉验证和错误分析脚本

## 后续可以继续优化

1. 对比 `bert-base-uncased`、`roberta-base`、`microsoft/deberta-v3-base`
2. 使用更稳的验证方案，比如按 `event` 分组验证
3. 在解释模块中引入注意力词、关键词权重或大语言模型生成解释
4. 增加 early stopping、warmup、weight decay 等训练策略
5. 在报告中加入 baseline 与 BERT 的误差对比和典型案例分析

## 小组协作建议

- 组长负责仓库维护、系统整合、README 和报告统稿
- 建模同学负责 baseline 和模型改进
- 解释模块同学负责判断依据生成和案例整理
- 每位成员都应直接提交代码或文档，保留清晰 commit 记录
