from typing import Any, Dict

import pytorch_lightning as pl

from ag_news_classifier.bert_model import AGNewsClassifier
from ag_news_classifier.config import Config
from ag_news_classifier.dataset import AGNewsDataModule


def train() -> Dict[str, Any]:
    cfg = Config()

    data_module = AGNewsDataModule(
        model_name=cfg.model.name,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )

    model = AGNewsClassifier(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
    )

    trainer = pl.Trainer(**vars(cfg.trainer))

    trainer.fit(model, data_module)

    test_results = trainer.test(model, data_module)

    return test_results[0] if test_results else {}


def validate(model_path: str) -> Dict[str, Any]:
    cfg = Config()
    data_module = AGNewsDataModule(
        model_name=cfg.model.name,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    model = AGNewsClassifier.load_from_checkpoint(model_path)
    trainer = pl.Trainer(**vars(cfg.trainer))
    val_results = trainer.validate(model, data_module)
    return val_results[0] if val_results else {}


def test(model_path: str) -> Dict[str, Any]:
    cfg = Config()
    data_module = AGNewsDataModule(
        model_name=cfg.model.name,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    model = AGNewsClassifier.load_from_checkpoint(model_path)
    trainer = pl.Trainer(**vars(cfg.trainer))
    test_results = trainer.test(model, data_module)
    return test_results[0] if test_results else {}
