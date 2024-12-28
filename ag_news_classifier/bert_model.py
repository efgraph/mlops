from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoModel, AutoTokenizer


class AGNewsClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        learning_rate: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, compute_on_cpu=True
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, compute_on_cpu=True
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, compute_on_cpu=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        accuracy = getattr(self, f"{stage}_accuracy")
        accuracy(logits, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True)

        return loss

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch):
        return self._shared_step(batch, "val")

    def test_step(self, batch):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def predict_text(self, text: str) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

            class_names = ["World", "Sports", "Business", "Science/Technology"]

            return {
                "class_name": class_names[predicted_class],
                "confidence": confidence,
            }
