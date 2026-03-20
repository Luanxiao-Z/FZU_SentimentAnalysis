# ==========================================
# 细粒度情感分析训练脚本（6 分类）
# 使用 BERT 对中文文本进行六类情感分类：悲伤、开心、生气、惊讶、恐惧、厌恶
# 支持从 data 目录加载 CSV，按 8:1:1 划分训练/验证/测试集，GPU 训练，输出完整评估指标
# ==========================================

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# 脚本所在目录与项目根目录（便于解析 data 与保存路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ------------------------------------------
# 配置参数（作为默认值，可被命令行覆盖）
# ------------------------------------------

NUM_LABELS = 6  # 分类类别数：6 种细粒度情感 (悲伤、开心、生气、惊讶、恐惧、厌恶)

DEFAULT_MAX_LENGTH = 128  # 文本最大长度，超过此长度的文本会被截断，不足的会 padding。BERT 最大支持 512
DEFAULT_BATCH_SIZE = 16  # 每批次训练的样本数。显存允许的情况下越大越好，可加速训练但会增加内存占用
DEFAULT_NUM_EPOCHS = 8  # 训练轮数，细粒度少数类往往在 3-4 轮后才开始收敛
DEFAULT_LEARNING_RATE = 2e-5  # 学习率：控制参数更新步长。BERT/Roberta 微调常用 2e-5 到 5e-5
DEFAULT_WEIGHT_DECAY = 0.02  # L2 正则，稍大一点抑制多数类过拟合
DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 更强的中文 RoBERTa-wwm 基座

RANDOM_STATE = 42  # 随机种子，保证划分可复现
TRAIN_RATIO, EVAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1  # 训练集:验证集:测试集 = 8:1:1

# 情感标签映射：数字编号 -> 情感名称，用于模型预测结果的解释和评估报告生成
EMOTION_NAMES = {0: "悲伤", 1: "开心", 2: "生气", 3: "惊讶", 4: "恐惧", 5: "厌恶"}
# 粗粒度情感映射：细粒度 -> 粗粒度 (用于后续可能的二分类任务)
COARSE_MAP = {0: "负面", 1: "正面", 2: "负面", 3: "中性", 4: "负面", 5: "负面"}
# 情感名称 -> 数字标签（用于部分数据集仅有 emotion_name 列的情况）
EMOTION_NAME_TO_ID = {v: k for k, v in EMOTION_NAMES.items()}


def parse_args():
    """解析命令行参数：数据文件名、模型与训练超参、输出目录。"""
    parser = argparse.ArgumentParser(description="细粒度情感分析模型训练")
    parser.add_argument(
        "--data",
        type=str,
        default="train_emotion.csv",
        help="数据文件名，置于 data 目录下。如 data.csv（约10万条）或 OCEMOTION.csv（约4万条）。建议先用 OCEMOTION 调参再用 data.csv 优化。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="模型与指标保存目录，默认: models/emotion_model",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"预训练 BERT 名称或路径，默认: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"训练轮数，默认: {DEFAULT_NUM_EPOCHS}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batch 大小，默认: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"学习率，默认: {DEFAULT_LEARNING_RATE}",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"序列最大长度，默认: {DEFAULT_MAX_LENGTH}",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"权重衰减（L2 正则），默认: {DEFAULT_WEIGHT_DECAY}",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing 系数（0 表示关闭，默认 0.1）",
    )
    parser.add_argument(
        "--no_class_weight",
        action="store_true",
        help="不使用类别权重（默认使用权重以缓解类别不平衡）",
    )
    parser.add_argument(
        "--boost_weak",
        type=float,
        default=1.0,
        help="对弱类(悲伤0、生气2)的类别权重再乘的系数，>1 表示更关注这两类，默认 1.0 不加重",
    )
    return parser.parse_args()


def load_and_prepare_data(data_filename):
    """
    从 data 目录加载 CSV，统一为 text + label 两列，并过滤无效行。
    支持列名：text/label（如 data.csv），或 emotion_name 映射为 label（如部分 OCEMOTION 格式）。
    """
    from pathlib import Path
    
    data_path = os.path.join(BASE_DIR, "data", data_filename)
    path_obj = Path(data_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    # Windows 下处理中文/长路径：使用 Path 对象和 open() 绕过限制
    try:
        # 方法 1: 使用 Path.resolve() 获取绝对路径并用 open() 读取
        resolved_path = path_obj.resolve()
        with open(resolved_path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
    except Exception as e1:
        try:
            # 方法 2: 尝试 GBK 编码
            with open(resolved_path, 'r', encoding='gbk') as f:
                df = pd.read_csv(f)
        except Exception as e2:
            try:
                # 方法 3: 使用 Windows 长路径前缀\\?\
                long_path_prefix = "\\\\?\\"
                long_path = long_path_prefix + str(resolved_path)
                with open(long_path, 'r', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            except Exception as e3:
                raise OSError(
                    f"无法读取文件 {data_path}\n"
                    f"可能原因：\n"
                    f"  1. 文件被 Excel/WPS 等程序占用\n"
                    f"  2. 文件路径过长或包含特殊字符\n"
                    f"  3. pandas 版本过旧（当前版本：{pd.__version__}）\n"
                    f"建议：关闭占用文件的程序，或升级 pandas: pip install --upgrade pandas"
                )
    # 统一文本列：部分数据集使用 sentence 列名
    if "text" not in df.columns and "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "text" not in df.columns:
        raise ValueError(f"CSV 需包含 text 列，当前列: {list(df.columns)}")

    # 统一标签列：优先使用数字 label，否则用 emotion_name 映射为 0~5
    if "label" not in df.columns:
        if "emotion_name" in df.columns:
            df["label"] = df["emotion_name"].map(EMOTION_NAME_TO_ID)
        else:
            raise ValueError(f"CSV 需包含 label 或 emotion_name 列，当前列: {list(df.columns)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])  # 丢弃缺失值
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(0, 5)].reset_index(drop=True)  # 只保留 6 类标签 0~5
    return df[["text", "label"]]


def split_811(df, random_state=RANDOM_STATE):
    """
    按 8:1:1 分层划分训练集、验证集、测试集。
    stratify 参数确保各类别比例在划分后保持一致，避免类别不平衡。
    返回 (train_df, eval_df, test_df)。
    """
    train_df, rest_df = train_test_split(
        df, test_size=(EVAL_RATIO + TEST_RATIO), random_state=random_state, stratify=df["label"]
    )
    eval_df, test_df = train_test_split(
        rest_df, test_size=TEST_RATIO / (EVAL_RATIO + TEST_RATIO), random_state=random_state, stratify=rest_df["label"]
    )
    return train_df, eval_df, test_df


def get_class_weights(train_labels, boost_weak=1.0):
    """
    根据训练集标签计算类别权重（逆频率平衡），用于不平衡数据。
    少数类获得更大权重，使损失函数更关注少数类。
    boost_weak: 对弱类(悲伤0、生气2)的权重再乘的系数，>1 表示更关注这两类。
    """
    classes = np.arange(NUM_LABELS)
    raw_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=np.array(train_labels)
    )
    # 平滑权重：避免极端权重导致梯度震荡
    smooth_weights = np.sqrt(raw_weights)
    smooth_weights = smooth_weights / smooth_weights.mean()
    if boost_weak != 1.0:
        for idx in (0, 2):  # 悲伤(0)、生气(2)
            smooth_weights[idx] *= boost_weak
        smooth_weights = smooth_weights / smooth_weights.mean()  # 再归一化，保持 loss 尺度
    return torch.tensor(smooth_weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """
    带类别权重的 Trainer，在计算损失时对少数类赋予更大权重，
    缓解「开心」等大类主导、「恐惧」等小类被忽略的问题。
    """

    def __init__(self, class_weights=None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights  # (num_labels,) 在 device 上后再用
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # 兼容新版 transformers 传入的 num_items_in_batch 等参数
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)
        else:
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    训练/验证阶段使用的指标。
    eval_pred: 包含 predictions(logits) 和 labels(真实标签) 的元组。
    返回：accuracy(整体准确率)、f1_macro(宏平均 F1，各类别 F1 的算术平均)、f1_weighted(按样本数加权)。
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)  # 取概率最大的类别作为预测结果
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def compute_full_metrics(labels, preds):
    """
    按需求计算并返回完整指标：
    - 准确率：预测正确的样本 / 总样本
    - 每类精确率 P = TP/(TP+FP)，召回率 R = TP/(TP+FN)，F1 = 2*P*R/(P+R)
    - 宏平均：各类指标的算术平均
    """
    n_classes = len(EMOTION_NAMES)
    accuracy = accuracy_score(labels, preds)

    # 每类 precision, recall, f1, support；不取 average 得到每类数组
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        labels, preds, labels=range(n_classes), average=None
    )
    # 宏平均：各类算术平均
    precision_macro = float(np.mean(precision_per))
    recall_macro = float(np.mean(recall_per))
    f1_macro = float(np.mean(f1_per))

    return {
        "accuracy": accuracy,
        "precision_per_class": precision_per.tolist(),
        "recall_per_class": recall_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "support_per_class": support_per.tolist(),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(BASE_DIR, "models", "emotion_model")
    os.makedirs(output_dir, exist_ok=True)

    # 将命令行参数与默认配置解耦，便于调参
    bert_model = args.bert_model or DEFAULT_MODEL_NAME
    num_epochs = args.epochs or DEFAULT_NUM_EPOCHS
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    learning_rate = args.lr or DEFAULT_LEARNING_RATE
    max_length = args.max_length or DEFAULT_MAX_LENGTH
    weight_decay = args.weight_decay or DEFAULT_WEIGHT_DECAY
    label_smoothing = float(getattr(args, "label_smoothing", 0.1))

    print("=" * 60)
    print("😊 细粒度情感分析系统")
    print("类别：悲伤(0) 开心(1) 生气(2) 惊讶(3) 恐惧(4) 厌恶(5)")
    print("=" * 60)

    # ------------------------------------------
    # 检查 GPU 设备
    # ------------------------------------------
    # 自动检测 CUDA 是否可用，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()  # 有 GPU 时启用混合精度，加速训练并减少显存占用
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------
    # 加载数据集（来自 data 目录，按 8:1:1 划分）
    # ------------------------------------------
    print(f"\n加载数据: {args.data}")
    df = load_and_prepare_data(args.data)
    train_df, eval_df, test_df = split_811(df)
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(eval_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    print("  训练集类别分布:")
    for i, name in EMOTION_NAMES.items():
        print(f"    {name}({i}): {(train_df['label'] == i).sum()} 条")

    # 将 DataFrame 转换为 Hugging Face Dataset 格式，便于后续处理
    # reset_index 确保索引连续，避免 Dataset 转换时出现警告
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # ------------------------------------------
    # 加载预训练 BERT 模型（6 分类）
    # ------------------------------------------
    print(f"\n加载模型: {bert_model}")
    print(f"输出类别数: {NUM_LABELS}")
    # 加载中文 BERT 分词器，用于将文本转换为模型输入
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载 BERT 序列分类模型；num_labels=6 表示最后全连接层输出 6 个 logits，对应 6 种情感
    # id2label/label2id：设置标签映射，方便推理时将预测的 ID 转换为情感名称
    model = BertForSequenceClassification.from_pretrained(
        bert_model,
        num_labels=NUM_LABELS,
        id2label=EMOTION_NAMES,
        label2id={v: k for k, v in EMOTION_NAMES.items()},
    )
    model = model.to(device)  # 将模型移动到 GPU/CPU

    # ------------------------------------------
    # 数据预处理：文本编码
    # ------------------------------------------
    def tokenize_function(examples):
        """
        对文本进行分词和编码。
        padding='max_length': 所有序列填充到 MAX_LENGTH 长度。
        truncation=True: 超过 MAX_LENGTH 的文本会被截断。
        return_tensors=None: 返回 Python 列表而非 Tensor，便于 Dataset 处理。
        """
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

    print("\n编码数据...")
    # 使用 map 函数批量处理数据集，应用分词函数
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 设置 Dataset 格式：只保留模型需要的列，type='torch' 转为 PyTorch Tensor
    cols = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=cols)
    eval_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    # ------------------------------------------
    # 训练参数配置
    # ------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,  # 模型与 checkpoint 保存目录
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,  # 每个设备的训练批大小
        per_device_eval_batch_size=batch_size,   # 每个设备的评估批大小
        warmup_ratio=0.1,      # 前 10% 步数线性增加学习率，有助于稳定训练
        weight_decay=weight_decay,     # L2 正则化，防止过拟合
        logging_dir=os.path.join(BASE_DIR, "logs"),
        logging_steps=50,      # 每 50 步记录一次日志
        evaluation_strategy="epoch", # 每个 epoch 结束后在验证集上评估（兼容旧版 transformers）
        save_strategy="epoch", # 每个 epoch 结束后保存 checkpoint
        load_best_model_at_end=True,   # 训练结束时加载验证集上最佳的模型
        metric_for_best_model="f1_macro",  # 用宏平均 F1 选最佳模型
        greater_is_better=True,
        save_total_limit=2,    # 最多保留 2 个 checkpoint，节省磁盘
        learning_rate=learning_rate,
        fp16=use_fp16,        # GPU 时启用混合精度
        report_to="none",     # 不向 wandb 等上报
    )

    # ------------------------------------------
    # 类别权重（不平衡数据时提升少数类影响力）
    # ------------------------------------------
    use_class_weight = not args.no_class_weight
    boost_weak = float(getattr(args, "boost_weak", 1.0))
    class_weights = None
    if use_class_weight:
        class_weights = get_class_weights(train_df["label"].tolist(), boost_weak=boost_weak)
        print(f"\n已启用类别权重（逆频率平衡），少数类将获得更大损失权重")
        if boost_weak != 1.0:
            print(f"  并对弱类 悲伤(0)、生气(2) 额外乘系数 {boost_weak}")

    # ------------------------------------------
    # 创建 Trainer 并开始训练
    # ------------------------------------------
    trainer_cls = WeightedTrainer if class_weights is not None or label_smoothing > 0 else Trainer
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 验证集指标连续 3 个 epoch 无提升则早停
    )
    if trainer_cls is WeightedTrainer:
        trainer_kw["class_weights"] = class_weights
        trainer_kw["label_smoothing"] = label_smoothing
    trainer = trainer_cls(**trainer_kw)

    print("\n开始训练...")
    trainer.train()

    # ------------------------------------------
    # 测试集详细评估（准确率、精确率、召回率、F1、宏平均）
    # ------------------------------------------
    print("\n" + "=" * 60)
    print("测试集评估指标")
    print("=" * 60)

    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = np.array(pred_output.label_ids)

    metrics = compute_full_metrics(labels, preds)

    print(f"\n准确率 (Accuracy): {metrics['accuracy']:.4f}  (预测正确的样本/总样本)")
    print("\n宏平均 (各类算术平均):")
    print(f"  精确率 (Precision): {metrics['precision_macro']:.4f}  [TP/(TP+FP)]")
    print(f"  召回率 (Recall):    {metrics['recall_macro']:.4f}  [TP/(TP+FN)]")
    print(f"  F1-Score:          {metrics['f1_macro']:.4f}  [2*P*R/(P+R)]")

    print("\n各类别 精确率 / 召回率 / F1-Score / 样本数:")
    target_names = list(EMOTION_NAMES.values())
    for i in range(NUM_LABELS):
        print(
            f"  {target_names[i]}({i}): "
            f"P={metrics['precision_per_class'][i]:.4f}  "
            f"R={metrics['recall_per_class'][i]:.4f}  "
            f"F1={metrics['f1_per_class'][i]:.4f}  "
            f"support={int(metrics['support_per_class'][i])}"
        )

    # 打印 sklearn 的 classification_report（与上述指标一致，便于对照）
    print("\nSklearn classification_report:")
    print(
        classification_report(
            labels, preds,
            target_names=target_names,
            digits=4,
        )
    )

    # 混淆矩阵：行=真实类别，列=预测类别，便于分析生气/悲伤等易混类
    cm = confusion_matrix(labels, preds, labels=range(NUM_LABELS))
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("      ", " ".join(f"{target_names[i]:>6}" for i in range(NUM_LABELS)))
    for i in range(NUM_LABELS):
        print(f" {target_names[i]:4s} ", " ".join(f"{cm[i, j]:6d}" for j in range(NUM_LABELS)))

    # 将指标与混淆矩阵保存为 JSON，便于汇报与存档
    metrics_save = {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_per_class": metrics["precision_per_class"],
        "recall_per_class": metrics["recall_per_class"],
        "f1_per_class": metrics["f1_per_class"],
        "support_per_class": metrics["support_per_class"],
        "confusion_matrix": cm.tolist(),
        "emotion_names": EMOTION_NAMES,
    }
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_save, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {metrics_path}")

    # ------------------------------------------
    # 保存模型和配置文件
    # ------------------------------------------
    print("\n保存模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # 保存自定义情感配置，便于推理时使用
    config = {
        "num_labels": NUM_LABELS,
        "emotion_names": EMOTION_NAMES,
        "coarse_map": COARSE_MAP,
        "id2label": EMOTION_NAMES,
        "label2id": {v: k for k, v in EMOTION_NAMES.items()},
    }
    with open(os.path.join(output_dir, "emotion_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"模型与配置已保存: {output_dir}")
    print("包含文件：")
    print("  - pytorch_model.bin (模型权重)")
    print("  - config.json (模型配置)")
    print("  - vocab.txt (词表)")
    print("  - tokenizer_config.json (分词器配置)")
    print("  - special_tokens_map.json (特殊 token 映射)")
    print("  - emotion_config.json (情感映射配置)")
    print("  - test_metrics.json (测试集评估指标)")


if __name__ == "__main__":
    main()
