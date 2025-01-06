## Классификатор новостных текстов

**Студент: Спиридонов Д.В., группа МОВС 2023**

## Содержание

- [Формулировка задачи](#формулировка-задачи)
- [Данные](#данные)
- [Подход к моделированию](#подход-к-моделированию)
- [Способ предсказания](#способ-предсказания)
- [Этап-2](#этап-2)
- [Этап-3](#этап-3)

## Формулировка задачи

**Что я делаю:** Цель проекта — разработка модели-классификатора, которая будет
оценивать ответы игроков на вопросы по темам ML/DL. Эти оценки станут основой
для игровых механик: правильные и более точные ответы будут давать игрокам
преимущество над другими участниками. Однако перед созданием финальной модели
для игровой задачи выбран подход с промежуточным этапом. Этот этап включает
тренировку и исследование модели на стандартном датасете

**Зачем это нужно:** Создать базовый рабочий прототип. Протестировать
архитектуру модели и методы обработки текста, а также избежать ошибок в
подготовке данных

## Данные

**Источники данных:**

На данном этапе мы используем классический датасет AG News, состоящий из
новостных текстов, которые нужно классифицировать по четырём категориям:

- World (Мир)
- Sports (Спорт)
- Business (Бизнес)
- Science/Technology (Наука/Технологии)

## Подход к моделированию

**Модель:**

- Базовая модель: `bert-base-uncased` (HuggingFace).
- Каждый элемент датасета состоит из:
  - текста - новостная статья или её фрагмент
  - метки - одна из четырёх категорий (0 — World, 1 — Sports, 2 — Business, 3 —
    Science/Technology)
- Данные сбалансированы — каждая категория содержит примерно одинаковое
  количество примеров

**Управление экспериментами:**

- Используем MLflow для отслеживания экспериментов
- DVC для версионирования данных и моделей

**Инструменты и библиотеки:**

- `transformers` (HuggingFace)
- `torch` (PyTorch)
- `datasets` (HuggingFace)
- `scikit-learn` для метрик

**Схема:**

    ┌─────────────────────────┐
    │      Исходные тексты    │
    └───────┬─────────────────┘
            │ Предобработка
            v
    ┌───────────────────────────┐
    │    Классификатор AG News  │
    └───────────┬───────────────┘
                │ Валидация и тестирование
                v
    ┌───────────────────────────────┐
    │ Классификация Вопрос–Ответ    │
    │        (оценка 0–3)           │
    └───────────┬───────────────────┘
                v
         Модель классификации

---

## Этап-2

1. Нарисуем структуру проекта командой

```
tree -a -I '__pycache__|*.pyc|.git|lightning_logs|*.idea|*logs|*data|*cache|*tmp'
```

```
.
├── ag_news_classifier
│   ├── bert_model.py
│   ├── dataset.py
│   ├── infer.py
│   ├── __init__.py
│   ├── logger_selector.py
│   ├── plotter.py
│   └── train.py
├── commands.py
├── conf
│   ├── config.yaml
│   ├── logging
│   │   └── logging.yaml
│   ├── model
│   │   └── model.yaml
│   ├── trainer
│   │   └── trainer.yaml
│   └── training
│       └── training.yaml
├── docker-compose.yaml
├── Dockerfile
├── .dvc
│   ├── config
│   └── .gitignore
├── .dvcignore
├── .gitignore
├── models
│   ├── .gitignore
│   └── model_val_loss=0.25.ckpt.dvc
├── plots
│   └── .gitignore
├── poetry.lock
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md

```

2. В корневой директории проекта создана отдельная директорию для сорсов
   ag_news_classifier
3. Убеждаемся что все зависимости зафиксированы в pyproject.toml, отсутствуют
   избыточные пакеты
4. Подготовим Dockerfile с валидным для проекта окружением, использовать его мы
   можем через

```
docker build -t ag_news_classifier .
docker run -d --name ag_news ag_news_classifier
docker exec -it ag_news python commands.py train
docker exec -it ag_news python commands.py infer --text "Hello world"
```

5. Проверим что `pre-commit run -a` выдает зеленый результат, присутствуют хуки
   black, isort, flake8. Также установим сначала `pre-commmit` хуки командой
   `pre-commit install`
6. Пока взял данные обучения не для диплома, а c huggingface датасет ag_news.
   Данных беру минимум для быстрого обучения, качество предсказаний пока
   неважно. Модель будет похожа на ту, что реализовал сейчас.
7. Запустить обучение (вместе с тестированием и валидацией) и инференс можем при
   помощи команд

```
poetry install
poetry shell
python commands.py train
python commands.py infer --text "Hello world"
```

## Этап-3

8. Вместо декоратора для hydra заиспользовал
   [Compose API](https://hydra.cc/docs/advanced/compose_api/)

9. Запускаем mlflow, указывая где хранятся метрики и артефакты

```
mlflow ui \
    --backend-store-uri file:./plots \
    --default-artifact-root file:./models \
    --host 0.0.0.0 \
    --port 5000
```

10. Мы логируем функцию потерь, точность, полноту и f1 метрику при помощи
    `torchmetrics`. Версию кода mlflow логирует автоматически
    (`Overview -> Source`). Для логирования гиперпараметров используем
    `logger.log_hyperparams`

11. Отрисовкой и сохранением графиков в формате `png` занимается
    `MetricPlotterCallback`, сохраняем в директорию `plots`

12. Хранилище S3 поднимаем локально через `docker-compose up`, по адресу
    `http://localhost:9001` создадим S3 бакет с названием `dvc-bucket` (креды
    admin/admin123)

- Инициализируем dvc и добавим ему поднятое хранилище S3 в remote

```
dvc init
dvc remote add -d myremote s3://dvc-bucket
dvc remote modify myremote endpointurl http://localhost:9000
dvc remote modify myremote access_key_id admin # логин
dvc remote modify myremote secret_access_key admin123 # пароль
```

- Добавим и запушим сохраненную модель в хранилище

```
dvc add models/model_val_loss\=0.22.ckpt
dvc push
```

- Теперь при желании мы можем ее достать из хранилища сделав

```
dvc pull
```

- Но в скачивание встроено в имеющиеся команды `train` и `infer`, поэтому
  произойдет автоматически, если передается аргумент `--dvc-pull`

```
python commands.py train --dvc-pull
python commands.py infer --dvc-pull --text "Hello world"
```
