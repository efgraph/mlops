from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class AGNewsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        train_size: int = 240,
        val_size: int = 100,
        test_size: int = 100,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        load_dataset("ag_news")
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None):
        dataset = load_dataset("ag_news")

        if stage == "fit" or stage is None:
            self.train_dataset = dataset["train"].select(range(self.train_size))
            self.val_dataset = dataset["test"].select(range(self.val_size))

        if stage == "test" or stage is None:
            self.test_dataset = dataset["test"].select(range(self.test_size))

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup(stage="fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup(stage="test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized["labels"] = torch.tensor(labels)
        return tokenized
