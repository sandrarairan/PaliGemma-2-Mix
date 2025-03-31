# PaliGemma 2 Mix - Mejores Prácticas

## Acerca de PaliGemma 2 Mix

PaliGemma 2 Mix es un modelo multimodal avanzado desarrollado por Google DeepMind que expande las capacidades de su predecesor. Este modelo integra procesamiento de imágenes y texto con capacidades mejoradas para tareas como:

- Generación de texto a partir de imágenes
- Respuesta a preguntas visuales
- Razonamiento multimodal complejo
- Comprensión de diagramas y gráficos
- Análisis de múltiples imágenes
- Edición y generación de imágenes basadas en texto

## Requisitos Técnicos

- Python 3.9+
- PyTorch 2.0+
- Espacio en disco: >15GB
- RAM recomendada: >16GB
- GPU: NVIDIA (8GB+ VRAM) para inferencia, 16GB+ para fine-tuning
- CUDA 11.7+ compatible

## Instalación

```bash
# Crear un entorno virtual
python -m venv paligemma_env
source paligemma_env/bin/activate  # En Windows: paligemma_env\Scripts\activate

# Instalar dependencias
pip install torch torchvision
pip install paligemma-2-mix

# Verificar instalación
python -c "from paligemma import PaliGemma2Mix; print('Instalación exitosa')"
```

## Uso Básico

```python
from paligemma import PaliGemma2Mix
from PIL import Image

# Cargar el modelo
model = PaliGemma2Mix.from_pretrained("google-deepmind/paligemma-2-mix")

# Preparar imagen
image = Image.open("example.jpg")

# Generar descripción
description = model.generate_text(images=[image], prompt="Describe esta imagen en detalle.")
print(description)

# Responder a una pregunta visual
answer = model.generate_text(images=[image], prompt="¿Qué elementos destacan en primer plano?")
print(answer)
```

## Mejores Prácticas

### 1. Preprocesamiento de Imágenes

- **Resolución óptima**: Las imágenes deben redimensionarse a 448×448 píxeles para un rendimiento óptimo.
- **Normalización**: Utilizar media=[0.5, 0.5, 0.5] y std=[0.5, 0.5, 0.5] para normalizar las imágenes.
- **Formatos recomendados**: JPEG, PNG, WebP.
- **Imágenes múltiples**: Para análisis multi-imagen, mantener un máximo de 8 imágenes por petición.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

processed_image = transform(image).unsqueeze(0)
```

### 2. Prompting Efectivo

- **Sea específico y claro**: "Describe los elementos arquitectónicos de este edificio" es mejor que "¿Qué ves?".
- **Utilice instrucciones paso a paso**: "Primero identifica el tema principal, luego describe los detalles".
- **Proporcione contexto**: "Esta es una imagen médica de un pulmón. Identifica posibles anomalías".
- **Formatos estructurados**: Solicite formatos específicos como JSON o listados cuando sea útil.

#### Ejemplos de Prompts Efectivos:

```
✅ "Analiza esta gráfica y resume las tendencias principales en forma de lista."
✅ "Identifica todos los objetos en la imagen y estima sus dimensiones relativas."
✅ "Compara las dos imágenes y señala las diferencias, enfocándote en los cambios de color."
```

### 3. Fine-tuning

- **Tamaño de dataset**: Mínimo recomendado: 1,000 ejemplos para tareas específicas.
- **Balanceo de datos**: Asegurar diversidad y balance en las imágenes y textos.
- **Parámetros recomendados**:
  - Learning rate: 1e-5 a 5e-5
  - Batch size: 4-16 dependiendo de la memoria disponible
  - Epochs: 3-5 para la mayoría de las tareas

```python
from paligemma.tuning import fine_tune

fine_tune(
    model=model,
    train_dataset=dataset,
    output_dir="./fine_tuned_model",
    learning_rate=2e-5,
    batch_size=8,
    num_epochs=3,
    save_steps=500
)
```

### 4. Optimización de Inferencia

- **Batch processing**: Procese múltiples imágenes en batch para mayor eficiencia.
- **Cuantización**: Utilice cuantización int8 para inferencia con recursos limitados.
- **Caching**: Implemente un sistema de caché para resultados frecuentes.
- **API vs. Local**: Para producción, considere usar la API de PaliGemma en lugar de despliegue local.

```python
# Ejemplo de batch processing
batch_images = [Image.open(f) for f in ["img1.jpg", "img2.jpg", "img3.jpg"]]
batch_prompts = ["Describe esta imagen.", "¿Qué ocurre en esta escena?", "Identifica los objetos."]

results = model.generate_batch(images=batch_images, prompts=batch_prompts)
```

### 5. Evaluación y Monitoreo

- **Métricas clave**: BLEU, ROUGE, CIDEr para tareas de generación; Accuracy, F1 para tareas clasificación.
- **Evaluación humana**: Implemente evaluaciones humanas periódicas para calidad subjetiva.
- **Monitoreo de latencia**: Mantenga registro de tiempos de respuesta en producción.
- **Control de calidad**: Establezca umbrales mínimos de confianza para respuestas.

```python
from paligemma.evaluation import evaluate_model

metrics = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    metrics=["bleu", "rouge", "accuracy"]
)
```


## Estructura del Repositorio

```
PaliGemma-2-Mix/
├── examples/               # Ejemplos de código para diferentes casos de uso
├── notebooks/              # Jupyter notebooks con tutoriales
├── scripts/                # Scripts útiles para procesamiento y evaluación
├── models/                 # Modelos fine-tuned para tareas específicas
├── data/                   # Datos de ejemplo y utilidades de preprocesamiento
└── docs/                   # Documentación adicional
```



## Recursos Adicionales

- [Documentación Oficial de PaliGemma]([https://github.com/google-deepmind/paligemma](https://developers.googleblog.com/en/introducing-paligemma-2-mix/)
- [huggingface](https://huggingface.co/google/paligemma2-3b-mix-224)
