# src/model.py
from transformers import AutoModelForSequenceClassification

def get_distilbert_classifier(model_name: str = "distilbert-base-uncased", num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )