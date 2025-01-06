from typing import Any, Dict

import pytorch_lightning as pl
from omegaconf import DictConfig

from ag_news_classifier.bert_model import AGNewsClassifier
from ag_news_classifier.dataset import AGNewsDataModule
from ag_news_classifier.logger_selector import get_logger
from ag_news_classifier.plotter import MetricPlotterCallback


def train(cfg: DictConfig) -> Dict[str, Any]:
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

    logger = get_logger(cfg.logging)

    logger.log_hyperparams(
        {
            "model_name": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
            "max_length": cfg.data.max_length,
            "num_workers": cfg.data.num_workers,
            "train_size": cfg.data.train_size,
            "val_size": cfg.data.val_size,
            "test_size": cfg.data.test_size,
        }
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./models",
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    plot_metrics_callback = MetricPlotterCallback()

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        strategy=cfg.trainer.strategy,
        logger=logger,
        callbacks=[checkpoint_callback, plot_metrics_callback],
    )

    trainer.fit(model, data_module)

    test_results = trainer.test(model, data_module)

    return test_results[0] if test_results else {}


def validate(model_path: str, cfg: DictConfig) -> Dict[str, Any]:
    data_module = AGNewsDataModule(
        model_name=cfg.model.name,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    model = AGNewsClassifier.load_from_checkpoint(model_path)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        strategy=cfg.trainer.strategy,
    )
    val_results = trainer.validate(model, data_module)
    return val_results[0] if val_results else {}


def test(model_path: str, cfg: DictConfig) -> Dict[str, Any]:
    data_module = AGNewsDataModule(
        model_name=cfg.model.name,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    model = AGNewsClassifier.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        strategy=cfg.trainer.strategy,
    )
    test_results = trainer.test(model, data_module)
    return test_results[0] if test_results else {}
