import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
import logging
import optuna
from transformers import get_scheduler, set_seed

import random, numpy as np, torch
#logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "f1_weigthed": f1_score(labels, predictions, average="weighted"),
            "f1_macro": f1_score(labels, predictions, average="macro"),
        }


from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_on_test(model, test_dataset, print_metrics=False):
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=64,
        report_to="none",
        logging_dir=None
    )

    trainer = Trainer(model=model, args=training_args)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    print(f"F1 macro en test: {f1_macro:.4f}")
    print(f"F1 promedio en test: {f1_weighted:.4f}")
   
    if print_metrics:
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds)
        print(report)
        print(cm)

    return f1_macro, f1_weighted


class OptunaCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        f1 = metrics.get("eval_f1_macro")
        if f1 is not None:
            self.trial.report(f1, step=state.epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments
import torch
import numpy as np

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def bert_training_optuna(
    train_dataset,
    eval_dataset,
    model,
    trial=None,
    epochs=3,
    batch_size=64,
    verbose=True,
    weight_decay=0.01,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    dropout=None,
    label_column="labels",
    metric_for_best_model="loss",
    warmup_ratio=0
):
    """
    Train a BERT model with optional frozen layers, weighted loss, early stopping, and regularization.

    Args:
        train_dataset (Dataset): Tokenized training dataset with labels.
        eval_dataset (Dataset): Tokenized validation dataset with labels.
        model (AutoModelForSequenceClassification): Pre-trained model.
        trial (optuna.Trial, optional): Current Optuna trial.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size per device.
        verbose (bool): Whether to print training progress.
        weight_decay (float): Weight decay for optimizer.
        learning_rate (float): Learning rate for optimizer.
        lr_scheduler_type (str): Scheduler type.
        dropout (float, optional): Override model dropout.
        label_column (str): Name of label column in dataset.
        metric_for_best_model (str): Metric to monitor for best model.
        freeze_layers (int): Number of encoder layers to freeze (from bottom).

    Returns:
        Tuple[Model, float]: Trained model and F1 macro score.
    """
    set_seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    model.to(device)

    # Override dropout if specified
    if dropout is not None:
        if hasattr(model, "dropout"):
            model.dropout.p = dropout
        if hasattr(model, "classifier") and hasattr(model.classifier, "dropout"):
            model.classifier.dropout.p = dropout

    # Compute class weights
    y_train = [int(x[label_column]) for x in train_dataset]
    class_labels = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=class_labels, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    training_args = TrainingArguments(
        output_dir="./transformer_classification",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=None,
        disable_tqdm=not verbose,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        save_total_limit=1,
        greater_is_better=False if metric_for_best_model == "loss" else True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if trial is not None:
        callbacks.append(OptunaCallback(trial))

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks
    )

    trainer.train()
    metrics = trainer.evaluate()
    f1_macro = metrics.get("eval_f1_macro")
    f1_weighted = metrics.get("eval_f1_weighted")
    return trainer.model, f1_macro, f1_weighted


def train_base_bert(train_dataset, eval_dataset, model, batch_size):
    """
    Fine-tune a BERT base model using Hugging Face defaults, only ensuring CUDA is used if available.

    Args:
        train_dataset: Tokenized training dataset with 'labels'
        eval_dataset: Tokenized validation dataset with 'labels'
        num_labels (int): Number of output classes

    Returns:
        model: Fine-tuned BERT model
    """
    from transformers.utils import logging
    logging.set_verbosity_info()

    model.to(device)
    training_args = TrainingArguments(
        output_dir="./transformer_classification",
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        report_to="none", 
        per_device_train_batch_size=batch_size,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset
    )
    print("Comenzando entrenamiento")
    trainer.train()
    return model



def dataframe_to_dataset(df, text_column: str, label_column: str, tokenizer, max_length: int = 128):
    from datasets import Dataset, disable_progress_bar, enable_progress_bar
    """Convert a pandas DataFrame to a tokenized Hugging Face dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing the text.
        label_column (str): Name of the column containing the label.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        max_length (int, optional): Maximum sequence length. Defaults to 512.

    Returns:
        datasets.Dataset: Tokenized dataset with 'labels' column.
    """
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=max_length)
    disable_progress_bar()
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    enable_progress_bar()
    return tokenized_dataset


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=class_weights.to(device))

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss



