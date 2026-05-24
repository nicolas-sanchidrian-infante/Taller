# Heart Disease Prediction Pipeline

Pipeline completo para desplegar una API de predicción de enfermedad cardíaca con Docker, Prometheus, Grafana y GitHub Actions.

## Arquitectura

- `api_heart.py`: API FastAPI con endpoints de salud, predicción y métricas.
- `heart_model_wrapper.py`: normaliza la entrada y llama al modelo.
- `heart_disease_model.joblib`: modelo serializado usado en producción.
- `docker-compose.yml`: levanta API, Prometheus y Grafana.
- `.github/workflows/ci.yml`: valida el proyecto en cada push o pull request.

## Requisitos

- Docker Desktop en ejecución.
- Puerto 8000 libre para la API.
- Puerto 9090 libre para Prometheus.
- Puerto 3000 libre para Grafana.

## Arranque local

```bash
docker compose up -d --build
```

Comprueba el estado con:

```bash
docker compose ps
```

## URLs

- API: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000
- UI: http://127.0.0.1:8000/ui

Credenciales de Grafana:

- Usuario: `admin`
- Contraseña: `admin`

## Ejemplo de predicción

### curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": 1,
    "chest_pain": 3,
    "rest_blood_pressure": 130,
    "cholesterol": 250,
    "fasting_blood_sugar": 0,
    "rest_ecg": 1,
    "max_heart_rate": 150,
    "exercise_angina": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "vessels": 0,
    "thal": 6
  }'
```

### Respuesta esperada

```json
{
  "prediction": 0,
  "prediction_label": "No Disease",
  "confidence": 0.59,
  "probabilities": {
    "no_disease": 0.59,
    "disease": 0.41
  },
  "risk_level": "Medium",
  "inference_time_ms": 9.05
}
```

## Métricas expuestas

La API publica métricas Prometheus en `/metrics`:

- `predictions_total`
- `prediction_duration_seconds`
- `active_predictions`
- `prediction_errors_total`

Prometheus ya está configurado para scrapear `app:8000`.

## Grafana

El dashboard se aprovisiona automáticamente desde:

- `grafana/provisioning/datasources/datasource.yml`
- `grafana/provisioning/dashboards/dashboards.yml`
- `grafana/dashboards/heart-disease-dashboard.json`


## CI

El workflow `.github/workflows/ci.yml`:

- compila los archivos Python,
- construye la imagen Docker,
- arranca el contenedor,
- valida `/health`,
- valida `/metrics`.

## Publicación de imagen

Publicar la imagen en GitHub Container Registry, usa el workflow adicional `docker-publish.yml`.

