from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "bert-base-uncased"
    num_classes: int = 4


@dataclass
class DataConfig:
    max_length: int = 256
    num_workers: int = 4
    train_size: int = 240
    val_size: int = 100
    test_size: int = 100


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_epochs: int = 1


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int = 1
    max_epochs: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    precision: str = "16-mixed"
    deterministic: bool = True
    strategy: str = "auto"


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    trainer: TrainerConfig = TrainerConfig()
