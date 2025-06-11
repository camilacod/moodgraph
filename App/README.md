# Detector de Emociones 🇪

Esta aplicación web utiliza el modelo `daveni/twitter-xlm-roberta-emotion-es` para detectar emociones en texto en español. La aplicación muestra las 3 emociones principales detectadas en el texto ingresado.

## Características

- Análisis de emociones en texto en español
- Visualización de las 3 emociones predominantes con porcentajes


## Requisitos

Los requisitos están especificados en el archivo `requirements.txt`.

## Configuración

Hay dos modos de funcionamiento:

### 1. Modo API remota (recomendado para desarrollo)

Para usar la API de Hugging Face (más rápido, sin requerir GPU local):

1. Crea un token de acceso en [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Configura la variable de entorno:

```bash
export HF_API_TOKEN=tu_token_aqui
```

### 2. Modo local

Si no se configura un token, el modelo se descargará y ejecutará localmente (≈1.2GB de RAM).

## Ejecución

Para ejecutar la aplicación:

```bash
cd app
uvicorn app:app --reload --port 8000
```

Luego abre http://localhost:8000 en tu navegador.

## API

La aplicación expone un endpoint POST:

```
POST /predict

{
  "text": "Tu texto para analizar aquí"
}
```

Respuesta:

```json
{
  "top_3": [
    {"label": "joy", "score": 0.9},
    {"label": "surprise", "score": 0.05},
    {"label": "other", "score": 0.03}
  ]
}
```

## Emociones detectadas

El modelo está entrenado para detectar 7 clases de emociones:
- anger (enojo)
- disgust (disgusto)
- fear (miedo)
- joy (alegría)
- sadness (tristeza)
- surprise (sorpresa)
- other (otra)
