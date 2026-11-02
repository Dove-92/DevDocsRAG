# -*- coding: utf-8 -*-
"""训练 DevDocsRAG 技术文档领域查询分类器(复用 QueryClassifier，独立目录，不覆盖旧模型)。"""
import os, sys, json, time

PROJECT_ROOT = r"E:\LLM_ubuntu\rag_qa_system"
sys.path.insert(0, PROJECT_ROOT)

from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from rag_qa.core.query_classifier import QueryClassifier

NEW_MODEL_PATH = os.path.join(PROJECT_ROOT, "rag_qa", "models", "bert_query_classifier_devdocs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rag_qa", "classify_data", "devdocs_classify", "bert_results_devdocs")
DATA_DIR = os.path.join(PROJECT_ROOT, "rag_qa", "classify_data", "devdocs_classify")


def load(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    return [d["query"] for d in data], [d["label"] for d in data]


print("== 训练前检查 ==")
qc = QueryClassifier(model_path=NEW_MODEL_PATH)  # 新路径不存在 -> 从 bert-base-chinese 初始化新模型
print("label_map:", qc.label_map)
print("device:", qc.device)
print("model 结构:", type(qc.model).__name__)
print("num_labels:", qc.model.config.num_labels)
print("tokenizer 类型:", type(qc.tokenizer).__name__)
print("hidden_size:", qc.model.config.hidden_size)

train_texts, train_labels = load(os.path.join(DATA_DIR, "train.jsonl"))
val_texts, val_labels = load(os.path.join(DATA_DIR, "val.jsonl"))
print(f"train: {len(train_texts)} 条 | val: {len(val_texts)} 条")
print("数据标签集合:", sorted(set(train_labels + val_labels)))
print("数据格式与 label_map 兼容:", set(train_labels + val_labels) <= set(qc.label_map.keys()))

train_enc, train_lbl = qc.preprocess_data(train_texts, train_labels)
val_enc, val_lbl = qc.preprocess_data(val_texts, val_labels)
train_ds = qc.create_dataset(train_enc, train_lbl)
val_ds = qc.create_dataset(val_enc, val_lbl)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=20,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    fp16=False,
    save_safetensors=False,
)

trainer = Trainer(
    model=qc.model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=qc.compute_metrics,
)

print("\n== 开始训练 ==")
t0 = time.time()
trainer.train()
print(f"训练耗时: {time.time() - t0:.1f}s")

print("\n== 保存模型 ==")
qc.save_model()
with open(os.path.join(NEW_MODEL_PATH, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(qc.label_map, f, ensure_ascii=False)
print(f"模型已保存至: {NEW_MODEL_PATH}")

# val 快速评估
val_pred = [qc.predict_category(t) for t in val_texts]
val_num = [qc.label_map[l] for l in val_labels]
pred_num = [qc.label_map[p] for p in val_pred]
print("val accuracy:", round(accuracy_score(val_num, pred_num), 4))
print("训练完成")
