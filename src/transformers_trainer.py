import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

from datasets import Dataset, DatasetDict


def prepare_hf_dataset(df, tokenizer, text_col='lyrics', label_col='label_vec', max_length=256):
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    dataset = Dataset.from_dict({
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'labels': labels
    })
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_micro': f1_score(labels, preds, average='micro')
    }


def train_bert(train_df, val_df, model_name='bert-base-uncased', output_dir='models/bert', epochs=3, batch_size=16, lr=2e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(train_df.iloc[0]['label_vec'])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type='multi_label_classification', num_labels=num_labels)

    train_ds = prepare_hf_dataset(train_df, tokenizer)
    val_ds = prepare_hf_dataset(val_df, tokenizer)
    ds = DatasetDict({'train': train_ds, 'validation': val_ds})

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy='epoch',
        learning_rate=lr,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    return output_dir