import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class MetricPlotterCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)
        self.epoch_counter = 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_counter += 1
        metrics_dict = trainer.callback_metrics

        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, float):
                value = metric_value
            elif hasattr(metric_value, "item"):
                value = metric_value.item()
            else:
                continue

            if any(
                metric_name.startswith(prefix) for prefix in ["train_", "val_", "test_"]
            ):
                self.history[metric_name].append(value)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.history:
            print("Nothing to plot")
            return

        os.makedirs("plots", exist_ok=True)

        mlflow_logger = None
        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            mlflow_logger = trainer.logger
            mlflow_run_id = mlflow_logger.run_id
        else:
            print("MLFlowLogger not found")

        for metric_name, values in self.history.items():
            epochs = range(1, len(values) + 1)
            plt.figure()
            plt.plot(epochs, values, marker="o", label=metric_name)
            plt.xlabel("epoch")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name}")
            plt.legend()

            plot_path = os.path.join("plots", f"{metric_name}.png")
            plt.savefig(plot_path)
            plt.close()

            if mlflow_logger is not None:
                mlflow_logger.experiment.log_artifact(mlflow_run_id, local_path=plot_path)
                print(f"Logged plot: {plot_path} as png")
