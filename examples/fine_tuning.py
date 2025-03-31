"""
Ejemplo de Fine-tuning de PaliGemma 2 Mix
========================================

Este script muestra cómo realizar fine-tuning del modelo PaliGemma 2 Mix
para tareas específicas como clasificación de imágenes médicas.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from paligemma import PaliGemma2Mix
from paligemma.tuning import fine_tune, prepare_model_for_tuning
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class MedicalImageDataset(Dataset):
    """Dataset personalizado para imágenes médicas con etiquetas"""
    
    def __init__(self, data_root, csv_file, transform=None):
        """
        Args:
            data_root (str): Directorio con las imágenes.
            csv_file (str): Ruta al archivo CSV con los nombres de archivo y etiquetas.
            transform (callable, optional): Transformación opcional a aplicar a las imágenes.
        """
        self.data_root = data_root
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Crear prompt para cada imagen
        self.data['prompt'] = "¿Qué patología muestra esta imagen médica?"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        prompt = self.data.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'prompt': prompt,
            'label': label
        }


def main(args):
    # Configurar transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Cargar datasets
    print(f"Cargando datos de {args.data_root}...")
    full_dataset = MedicalImageDataset(
        data_root=args.data_root,
        csv_file=args.csv_file,
        transform=transform
    )
    
    # Dividir en entrenamiento y validación
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=[full_dataset[i]['label'] for i in range(len(full_dataset))]
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del dataset de validación: {len(val_dataset)}")
    
    # Cargar modelo
    print("Cargando modelo PaliGemma 2 Mix...")
    model = PaliGemma2Mix.from_pretrained("google-deepmind/paligemma-2-mix")
    
    # Preparar modelo para fine-tuning
    model = prepare_model_for_tuning(model)
    
    # Configurar parámetros de fine-tuning
    training_args = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "logging_steps": 50,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "fp16": torch.cuda.is_available(),
    }
    
    # Realizar fine-tuning
    print(f"Iniciando fine-tuning por {args.epochs} épocas...")
    fine_tune_results = fine_tune(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        **training_args
    )
    
    # Guardar resultados
    print(f"Fine-tuning completado. Guardando resultados en {args.output_dir}")
    fine_tune_results.save_metrics("train", os.path.join(args.output_dir, "train_results.json"))
    fine_tune_results.save_metrics("eval", os.path.join(args.output_dir, "eval_results.json"))
    
    # Visualizar resultados
    if args.show_results:
        train_loss = fine_tune_results.state.log_history
        
        # Extraer pérdidas y exactitud
        epochs = list(range(1, args.epochs + 1))
        train_losses = [h.get('loss') for h in train_loss if 'loss' in h]
        eval_losses = [h.get('eval_loss') for h in train_loss if 'eval_loss' in h]
        eval_acc = [h.get('eval_accuracy') for h in train_loss if 'eval_accuracy' in h]
        
        # Graficar resultados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(epochs, train_losses, 'b-', label='Entrenamiento')
        ax1.plot(epochs, eval_losses, 'r-', label='Validación')
        ax1.set_title('Pérdida durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        
        ax2.plot(epochs, eval_acc, 'g-')
        ax2.set_title('Exactitud de validación')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Exactitud')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_results.png'))
        plt.show()
    
    print("Proceso completado con éxito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning de PaliGemma 2 Mix")
    parser.add_argument("--data_root", type=str, required=True, help="Carpeta raíz con las imágenes")
    parser.add_argument("--csv_file", type=str, required=True, help="Archivo CSV con nombres de archivo y etiquetas")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directorio para guardar el modelo")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Tasa de aprendizaje")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño del batch")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--show_results", action="store_true", help="Mostrar gráficos de resultados al finalizar")
    
    args = parser.parse_args()
    main(args)
