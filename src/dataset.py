# src/dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer

def get_sst2_data(model_name: str = "distilbert-base-uncased", max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset("stanfordnlp/sst2")
    
    def tokenize_batch(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",  # Añadido para igualar dimensiones
            max_length=max_length,
        )
    
    tokenized_datasets = raw_datasets.map(tokenize_batch, batched=True)
    cols_to_remove = [c for c in ["sentence", "idx"] if c in tokenized_datasets["train"].column_names]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets, tokenizer