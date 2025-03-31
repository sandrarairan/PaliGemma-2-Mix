"""
Ejemplo básico de uso de PaliGemma 2 Mix
=======================================

Este script muestra el uso básico del modelo PaliGemma 2 Mix para varias tareas
comunes como descripción de imágenes, responder preguntas visuales y análisis
de elementos en imágenes.
"""

import torch
from paligemma import PaliGemma2Mix
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

def main(args):
    # Cargar el modelo
    print("Cargando modelo PaliGemma 2 Mix...")
    model = PaliGemma2Mix.from_pretrained("google-deepmind/paligemma-2-mix")
    
    # Mover a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Modelo cargado y movido a {device}")

    # Cargar imagen
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {args.image_path}")
    
    image = Image.open(args.image_path)
    
    # Mostrar imagen si se especificó
    if args.show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Imagen de entrada")
        plt.show()
    
    # Ejecutar diferentes tareas según se especifique
    if args.task == "describe":
        prompt = "Describe esta imagen en detalle."
        print("\nGenerando descripción detallada de la imagen...")
    
    elif args.task == "question":
        prompt = args.question if args.question else "¿Qué elementos destacan en esta imagen?"
        print(f"\nRespondiendo a la pregunta: {prompt}")
    
    elif args.task == "analyze":
        prompt = "Analiza esta imagen e identifica los objetos principales, sus colores y sus posiciones relativas."
        print("\nAnalizando elementos de la imagen...")
        
    elif args.task == "custom":
        if not args.prompt:
            raise ValueError("Se debe proporcionar un prompt personalizado con --prompt cuando se usa --task=custom")
        prompt = args.prompt
        print(f"\nEjecutando prompt personalizado: {prompt}")
    
    else:
        raise ValueError(f"Tarea no reconocida: {args.task}")
    
    # Generación de texto
    with torch.no_grad():
        response = model.generate_text(
            images=[image],
            prompt=prompt,
            max_length=args.max_tokens,
            temperature=args.temperature
        )
    
    print("\n--- Respuesta de PaliGemma 2 Mix ---")
    print(response)
    print("-----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejemplo de uso de PaliGemma 2 Mix")
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen a analizar")
    parser.add_argument("--task", type=str, default="describe", 
                        choices=["describe", "question", "analyze", "custom"],
                        help="Tarea a realizar: describe, question, analyze, o custom")
    parser.add_argument("--question", type=str, help="Pregunta específica para la tarea 'question'")
    parser.add_argument("--prompt", type=str, help="Prompt personalizado para la tarea 'custom'")
    parser.add_argument("--max_tokens", type=int, default=256, help="Número máximo de tokens a generar")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura para la generación de texto")
    parser.add_argument("--show_image", action="store_true", help="Mostrar la imagen usando matplotlib")
    
    args = parser.parse_args()
    main(args)
