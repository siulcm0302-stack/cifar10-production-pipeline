"""
finetune.py - Script de Fine-Tuning con QLoRA (PEFT + BitsAndBytes)
Demuestra adaptación eficiente de modelos ligeros para tu portafolio AI Engineering.
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def run_qlora_training():
    model_id = "HuggingFaceTB/SmolLM-136M"  # Cambiar por Llama-3-8B / Phi-3 si hay VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configuración de cuantización 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Cargando modelo base con cuantización QLoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # Configuración LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset de prueba / dominio propio
    data = {"text": ["### Contexto: PyTorch\n### Pregunta: ¿Qué es?\n### Respuesta: Un framework de tensores."]}
    from datasets import Dataset
    dataset = Dataset.from_dict(data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./qlora_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=2,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Iniciando entrenamiento QLoRA de prueba...")
    trainer.train()
    print("¡Fine-tuning QLoRA finalizado con éxito!")

if __name__ == "__main__":
    run_qlora_training()