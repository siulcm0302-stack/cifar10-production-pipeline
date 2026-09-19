# train.py
from src.dataset import get_sst2_data
from src.model import get_distilbert_classifier
from src.trainer import get_trainer

def main():
    print("Cargando y tokenizando dataset...")
    datasets, tokenizer = get_sst2_data()
    
    print("Inicializando modelo...")
    model = get_distilbert_classifier(num_labels=2)
    
    print("Configurando trainer...")
    trainer = get_trainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )
    
    print("Iniciando entrenamiento...")
    trainer.train()
    
    print("Guardando artefactos finales...")
    trainer.save_model("./models/distilbert-sst2")
    tokenizer.save_pretrained("./models/distilbert-sst2")
    print("¡Entrenamiento completado!")

if __name__ == "__main__":
    main()