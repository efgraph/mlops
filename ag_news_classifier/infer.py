from typing import Any, Dict

from ag_news_classifier.bert_model import AGNewsClassifier


def infer(model_path: str, text: str) -> Dict[str, Any]:
    model = AGNewsClassifier.load_from_checkpoint(model_path)
    model.eval()

    prediction = model.predict_text(text)

    return {
        "text": text,
        "predicted_class": prediction["class_name"],
        "confidence": prediction["confidence"],
    }
