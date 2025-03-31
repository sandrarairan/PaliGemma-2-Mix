# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto PaliGemma 2 Mix! Esta guía te ayudará a entender cómo puedes participar y aportar al desarrollo de esta herramienta.

## Código de Conducta

Este proyecto se adhiere a un código de conducta para crear un ambiente inclusivo y respetuoso. Se espera que todos los contribuyentes cumplan con nuestro código de conducta en todas las interacciones del proyecto.

## ¿Cómo puedo contribuir?

Hay varias formas de contribuir al proyecto:

### 1. Reportar bugs

Si encuentras un bug:
- Verifica primero que el problema no haya sido reportado ya en los issues.
- Usa el template de bug para crear un nuevo issue.
- Incluye pasos detallados para reproducir el problema.
- Proporciona capturas de pantalla si es posible.
- Menciona tu entorno (sistema operativo, versión de Python, etc.).

### 2. Sugerir mejoras o nuevas características

Las sugerencias son bienvenidas:
- Verifica que tu idea no esté ya propuesta.
- Crea un nuevo issue usando el template de feature request.
- Describe claramente qué te gustaría ver implementado y por qué.
- Si es posible, sugiere cómo podría implementarse.

### 3. Contribuir con código

Para contribuir con código:

1. **Fork el repositorio**
2. **Crea una nueva rama** para tu característica:
   ```
   git checkout -b feature/nombre-descriptivo
   ```
3. **Realiza tus cambios** siguiendo las convenciones de código
4. **Ejecuta las pruebas** para asegurarte de que todo funciona
5. **Commit tus cambios**:
   ```
   git commit -m "Descripción clara del cambio"
   ```
6. **Push a tu rama**:
   ```
   git push origin feature/nombre-descriptivo
   ```
7. **Crea un Pull Request** describiendo tus cambios

### 4. Mejorar la documentación

La documentación es crucial:
- Corrige errores en la documentación existente.
- Añade ejemplos o tutoriales.
- Mejora las explicaciones.
- Traduce la documentación a otros idiomas.

### 5. Compartir casos de uso

Si has utilizado PaliGemma 2 Mix en un proyecto interesante:
- Crea un notebook detallando tu caso de uso.
- Comparte tus resultados y aprendizajes.
- Contribuye con ejemplos reales a la carpeta `/examples`.

## Convenciones de código

- Sigue las convenciones de PEP 8 para código Python.
- Utiliza nombres descriptivos para variables y funciones.
- Incluye docstrings para todas las funciones y clases.
- Comenta el código cuando sea necesario para explicar decisiones de implementación.
- Escribe tests para la nueva funcionalidad.

## Proceso de Pull Request

1. Un mantenedor revisará tu PR lo antes posible.
2. Es posible que se soliciten cambios o mejoras.
3. Una vez aprobado, un mantenedor hará merge de tu PR.
4. Tu contribución será reconocida en los créditos del proyecto.

## Estructura del proyecto

Para contribuir efectivamente, es importante entender la estructura del proyecto:

```
PaliGemma-2-Mix/
├── examples/               # Ejemplos de código
├── notebooks/              # Tutoriales y demostraciones
├── scripts/                # Scripts útiles
├── models/                 # Modelos fine-tuned
├── data/                   # Datos de ejemplo
└── docs/                   # Documentación
```

## Configuración del entorno de desarrollo

Para configurar tu entorno:

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/PaliGemma-2-Mix.git
cd PaliGemma-2-Mix

# Crear un entorno virtual
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest
```

## ¿Necesitas ayuda?

Si necesitas ayuda para contribuir:
- Revisa la documentación existente.
- Crea un issue con tu pregunta.
- Únete a nuestro canal de Discord para discusión en tiempo real.

¡Gracias por tu contribución!
