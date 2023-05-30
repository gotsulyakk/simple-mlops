import itertools
import os

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def get_all_tokens_and_ner_tags(directory):
    data_frames = [
        get_tokens_and_ner_tags(os.path.join(directory, filename))
        for filename in os.listdir(directory)
    ]
    return pd.concat(data_frames).reset_index(drop=True)


def get_tokens_and_ner_tags(filename):
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        split_list = [
            list(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x
        ]
        tokens = [[x.split("\t")[0] for x in y] for y in split_list]
        entities = [[x.split("\t")[1][:-1] for x in y] for y in split_list]
    return pd.DataFrame({"tokens": tokens, "ner_tags": entities})


def get_un_token_dataset(directory):
    return Dataset.from_pandas(get_all_tokens_and_ner_tags(directory))


def tokenize_and_align_labels(examples, tokenizer, label_encoding_dict):
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def train():
    MLFLOW_TRACKING_URI = "sqlite:///mlflow_data/mlflow.db"
    EXPERIMENT_NAME = "distilbert-un-ner"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Define data paths
    TRAIN_DATA_DIR = "./data/train"
    TEST_DATA_DIR = "./data/test"

    # Define labels
    LABEL_LIST = [
        "O",
        "B-MISC",
        "I-MISC",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
    ]
    LABELS_ENCODING_DICT = {
        "I-PRG": 2,
        "I-I-MISC": 2,
        "I-OR": 6,
        "O": 0,
        "I-": 0,
        "VMISC": 0,
        "B-PER": 3,
        "I-PER": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-LOC": 7,
        "I-LOC": 8,
        "B-MISC": 1,
        "I-MISC": 2,
    }

    # Load tokenizer
    task = "ner"
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Prepare datasets
    train_dataset = get_un_token_dataset(TRAIN_DATA_DIR)
    test_dataset = get_un_token_dataset(TEST_DATA_DIR)

    train_tokenized_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": LABELS_ENCODING_DICT},
    )
    test_tokenized_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": LABELS_ENCODING_DICT},
    )

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(LABEL_LIST)
    )

    # Define training arguments
    args = TrainingArguments(
        f"test-{task}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=1e-5,
    )

    # Define data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, LABEL_LIST),
    )

    # Train model and log to MLflow
    with mlflow.start_run():
        mlflow.log_param("train_data", TRAIN_DATA_DIR)
        mlflow.log_param("test_data", TEST_DATA_DIR)

        trainer.train()

        metrics = trainer.evaluate()

        mlflow.log_metrics(metrics)

        components = {"model": trainer.model, "tokenizer": tokenizer}
        model_info = mlflow.transformers.log_model(
            transformers_model=components, artifact_path="model", task="ner"
        )

        print(model_info)

        mlflow.end_run()


if __name__ == "__main__":
    train()
