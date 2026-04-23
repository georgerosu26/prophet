# Prophet API Service

FastAPI service that trains and predicts per-store forecasts using `facebook/prophet`.

## Endpoints

- `GET /health`
- `POST /train/{store_id}`
- `POST /predict/{store_id}`
- `POST /train-predict/{store_id}`
- `POST /backtest/{store_id}`
- `GET /model-status/{store_id}`

## Auth

Set `PROPHET_API_TOKEN` and send it as `x-api-token` header.

## Environment Variables

- `PROPHET_API_TOKEN` (optional but recommended)
- `MODEL_DIR` (default `/app/models`)
- `DEFAULT_HORIZON_DAYS` (default `91`)
- `MAX_HORIZON_DAYS` (default `180`)
- `MIN_TRAIN_ROWS` (default `60`)

## Base44 Integration

Use Base44 function `runProphetForecast` with env vars:

- `PROPHET_API_URL`
- `PROPHET_API_TOKEN`
