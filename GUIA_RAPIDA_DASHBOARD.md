# Guia Rapida - Dashboard Final

Esta guia explica como ejecutar y usar el dashboard final ubicado en `src/dashboard_layer/app.py`.

## 1) Ejecutar la app

```powershell
conda activate cmapss
streamlit run src/dashboard_layer/app.py
```

## 2) Que modos de entrada tiene

1. `Manual`
- Selecciona `Dataset`, `Unit ID`, `Cycle`.
- Ajusta `op_setting_1..3`.
- Ajusta `sensor_1..21`.
- Click en `Run Prediction`.

2. `CSV/TXT`
- Puedes subir:
  - CSV con columnas requeridas, o
  - TXT crudo NASA (`train_*.txt` / `test_*.txt`).
- Si subes TXT, selecciona dataset (FD001..FD004) cuando corresponda.
- Click en `Run Prediction`.

## 3) Columnas requeridas para CSV

- `dataset_id`
- `unit_id`
- `cycle`
- `op_setting_1`, `op_setting_2`, `op_setting_3`
- `sensor_1` ... `sensor_21`

## 4) Como leer las pestanas

1. `Resumen`
- KPI de RUL, banda de confianza, nivel y score de riesgo.

2. `Detalle`
- Contexto de entrada y salida completa.
- Rationale de decision.

3. `Historico`
- Tendencia de degradacion por ciclo (`rul_est`).

4. `Recomendaciones`
- Accion principal, alternativas y prioridad.
- Resumen de evidencia (si aplica).

5. `Escenarios`
- Prompt what-if en lenguaje natural.
- Propuesta de cambios (`change_summary`).
- Comparacion baseline vs scenario.

6. `Trazabilidad`
- `audit_record_id`
- `service_status`
- `model_version`
- `timestamp`

## 5) Estados de la app

- `sin_datos`: aun no hay prediccion.
- `loading`: pipeline en ejecucion.
- `ok`: prediccion lista.
- `error_validacion`: entrada invalida.
- `degraded`: servicio no disponible temporalmente.

## 6) Solucion rapida de problemas

1. Error de columnas CSV:
- Verifica nombres exactos de columnas y tipos numericos.

2. Error en TXT NASA:
- Revisa que tenga al menos 26 columnas separadas por espacios.

3. Estado `degraded`:
- Verifica que esten accesibles `src/agent_layer` y `src/predictive_layer`.
- Revisa artefactos de modelos en `out/predictive_layer/`.

4. Error al iniciar Streamlit:
- Confirma dependencias:
```powershell
pip install -r requirements.txt
```

## 7) Configurar GEMINI_API_KEY (opcional)

Solo necesario si habilitas funciones LLM reales en capa 2.
La capa 2 usa el SDK oficial `google-genai` (ya incluido en dependencias del proyecto).

Local (PowerShell, sesion actual):
```powershell
$env:GEMINI_API_KEY="tu_api_key_aqui"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

Local persistente:
```powershell
setx GEMINI_API_KEY "tu_api_key_aqui"
setx GEMINI_MODEL "gemini-2.5-flash"
```

Streamlit Cloud:
1. `Settings` -> `Secrets`
2. Agregar:
```toml
GEMINI_API_KEY = "tu_api_key_aqui"
GEMINI_MODEL = "gemini-2.5-flash"
```
3. Guardar y redeploy.

Si `GEMINI_MODEL` no existe o fue retirado, el sistema intenta fallback automatico a:
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-3-flash-preview`
