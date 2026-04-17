# CMAPSS PHM App (NASA C-MAPSS)

Proyecto de mantenimiento predictivo para motores turbofan usando NASA C-MAPSS.

La app esta organizada por capas:
1. `predictive_layer`: pronostico de RUL.
2. `agent_layer`: logica de riesgo y recomendaciones.
3. `dashboard_layer`: interfaz final para usuario.

## Objetivo del proyecto

Construir una app funcional de PHM (Prognostics and Health Management) que:
- reciba series multivariadas por motor/ciclo,
- estime RUL,
- clasifique nivel de riesgo,
- emita recomendaciones operativas con trazabilidad.

## Arquitectura actual

### Capa 1: Predictive Layer
- Codigo: `src/predictive_layer/`
- Runner: `src/run_plan3_predictive_layer.py`
- Salidas: `out/predictive_layer/`
- Incluye entrenamiento multi-modelo (RF, GB, LSTM, GRU), seleccion de champion y contrato de inferencia.

### Capa 2: Agent Layer
- Codigo: `src/agent_layer/`
- Runner: `src/run_plan4_agent_layer.py`
- Salidas: `out/agent_layer/`
- Orquesta riesgo y recomendaciones a partir de la salida de `predictive_layer`.

### Capa 3: Dashboard Layer (final)
- Codigo: `src/dashboard_layer/`
- Runner: `src/run_plan5_dashboard_layer.py`
- Salidas: `out/dashboard_layer/`
- Dashboard final integrado con capas 1 y 2.

### Dashboard mock (sandbox)
- Codigo legacy/sandbox: `dashboard/`
- Uso: prototipado UX y pruebas rapidas.
- Nota: no es la ruta final de producto.

## Estructura del repositorio

- `data/`: archivos C-MAPSS (`train_*.txt`, `test_*.txt`, `RUL_*.txt`)
- `doc/`: referencias tecnicas
- `src/`: implementacion por capas + runners de planes
- `out/`: artefactos de ejecucion por plan/capa
- `fig/`: figuras de EDA
- `plan*.txt`: planes de trabajo ejecutables

## Requisitos

- Conda (Miniconda o Anaconda)
- Python 3.11 recomendado
- Dataset NASA C-MAPSS en `data/`

Dependencias usadas en el proyecto (capas 1, 2 y 3):
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `openpyxl`
- `streamlit`
- `plotly`

## Setup rapido

Opcion recomendada (entorno completo reproducible):

```powershell
conda env create -f environment.yml
conda activate cmapss
```

Opcion manual:

```powershell
conda create -n cmapss python=3.11 -y
conda activate cmapss
conda install -y -c conda-forge numpy pandas scipy matplotlib scikit-learn joblib openpyxl pyarrow streamlit plotly
pip install -r requirements.txt
```

Verificacion rapida del entorno:

```powershell
python -c "import numpy,pandas,scipy,matplotlib,sklearn,joblib,openpyxl,streamlit,plotly; print('env ok')"
```

Nota sobre `requirements.txt`:
- `requirements.txt` cubre runtime del dashboard final (incluye `scikit-learn` y `joblib` para inferencia).
- Para ejecutar todo el pipeline (EDA, entrenamiento, evaluacion), usa `environment.yml` o instala el stack cientifico completo con conda.

## Ejecucion por planes

```powershell
conda run -n cmapss python src/run_plan1_eda.py
conda run -n cmapss python src/run_plan2_research.py
conda run -n cmapss python src/run_plan3_predictive_layer.py
conda run -n cmapss python src/run_plan4_agent_layer.py
conda run -n cmapss python src/run_plan5_dashboard_layer.py
```

## Ejecutar dashboard final

```powershell
conda activate cmapss
streamlit run src/dashboard_layer/app.py
```

Guia rapida de uso:
- Ver `GUIA_RAPIDA_DASHBOARD.md`.

## Despliegue en Streamlit Cloud

- Repository: este repositorio
- Branch: `main`
- Main file path: `src/dashboard_layer/app.py`

## Configuracion de API Key (Gemini)

Si habilitas LLM real en `agent_layer`, usa `GEMINI_API_KEY` como variable de entorno.

### Local (PowerShell)

Temporal (solo sesion actual):

```powershell
$env:GEMINI_API_KEY="tu_api_key_aqui"
```

Persistente para usuario actual:

```powershell
setx GEMINI_API_KEY "tu_api_key_aqui"
```

Luego abre una nueva terminal para que tome el valor persistente.

### Streamlit Cloud

1. Entra a la app en Streamlit Cloud.
2. Ve a `Settings` -> `Secrets`.
3. Agrega:

```toml
GEMINI_API_KEY = "tu_api_key_aqui"
```

4. Guarda y reinicia/redeploy la app.

Regla de seguridad:
- Nunca hardcodear API keys en codigo o repositorio.

## Estado actual

- Capa 1: funcional y con artefactos de evaluacion.
- Capa 2: funcional en flujo base (riesgo + recomendaciones + auditoria).
- Capa 3 final: implementada en `src/dashboard_layer`.
- Pendiente de cierre final:
  - smoke en nube,
  - endurecimiento de pruebas automaticas,
  - multimodal y LLM real (si se decide activar).
