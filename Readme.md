# Memoria para asistentes conversacionales

## Introducción

En un futuro cercano, los asistentes conversacionales correrán directamente *en tu celular*: modelos de LLMs pequeños, eficientes y capaces de recordar tus hábitos, tus gustos y el contexto de tus últimas conversaciones. Para que ese tipo de agentes sea posible, necesitan **módulos de memoria optimizados**: rápidos, baratos de ejecutar y sin depender de modelos gigantes. Este track propone justamente eso: **explorar cómo diseñar el mejor sistema de memoria para agentes conversacionales usando modelos pequeños de LLMs**, comparando distintas estrategias y evaluando su eficiencia y calidad.

Este repositorio acompaña el track de NLP del Investigathon de YHat y plantea el desafío de ampliar un asistente conversacional con un módulo de memoria.  
Incluimos una referencia básica basada en un *semantic retriever*, que servirá como baseline.

Un **retriever semántico** o **RAG** puede pensarse como una función que, dada una consulta `Q` y un conjunto de documentos `D`, calcula un embedding para cada documento y devuelve los **top-k documentos más relevantes** según su similitud con la consulta. 

La utilización de este proyecto es completamente opcional: pueden usarlo tal cual, adaptarlo o simplemente tomarlo como fuente de inspiración.

> **Nota:** la explicación completa del benchmark, el formato de las instancias y los criterios de evaluación (incluyendo cómo evaluamos correctitud y memoria) está en `benchmark_explanation.md`.


## Estructura del proyecto

La carpeta principal es `src`. Allí van a encontrar:

- **`models`**: implementaciones de referencia. `LiteLLM` simplifica la prueba de múltiples APIs al unificar su interfaz. Si trabajan con modelos de Hugging Face sugerimos Qwen3, que ofrece buen *reasoning* y soporte de *tools*; por eso incluimos `QwenModel`. Dependiendo del hardware y la experiencia, también pueden evaluar vLLM. En esta demo vamos a usar `ollama` para el servidor y `LiteLLM` como cliente unificado (todo esto se va a entender mas adelante). 
- **`agents`**: distintos agentes ya configurados. `JudgeAgent` evalúa si la respuesta es correcta. `RAGAgent` implementa el módulo de RAG que usamos para el benchmark.
- **`datasets`**: utilidades para cargar y representar el benchmark. Incluye la clase `LongMemEvalInstance`, alineada con la definición del paper. Pueden no usarla, o usarla simplemente como inspiracion.

```python
def instance_from_row(self, row):
    return LongMemEvalInstance(
        question_id=row["question_id"],
        question=row["question"],
        sessions=[
            Session(session_id=session_id, date=date, messages=messages)
            for session_id, date, messages in zip(
                row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
            )
        ],
        t_question=row["question_date"],
        answer=row["answer"],
    )
```

En `config` pueden definir qué modelo responde, qué modelo actúa como juez y otros parámetros. Los scripts principales (`main.py`, `run_evaluation.py`, `run_held_out.py`) implementan el pipeline experimental directamente.


## Setup

Recomendamos utilizar `uv` para gestionar el entorno. 

### Instalacion de uv

curl -LsSf https://astral.sh/uv/install.sh | sh


### Instalación de dependencias

Una vez instalado `uv`, sincronizá las dependencias:

```sh
uv sync
```

### Descarga de datasets

Con el entorno configurado, descargá todos los datasets (LongMemEval original + Investigathon) con un solo comando:

```sh
uv run scripts/download_dataset.py
```

Alternativamente, activá el entorno virtual y ejecutá el script manualmente:

```sh
source .venv/bin/activate
python scripts/download_dataset.py
```

Este script descargará automáticamente:

#### Dataset LongMemEval original (desde HuggingFace)
- **longmemeval_oracle.json** - Versión original del benchmark
- **longmemeval_s_cleaned.json** - Versión limpia del benchmark

#### Dataset Investigathon (desde Google Drive)
- **Investigathon_LLMTrack_Evaluation_oracle.json** (6.1 MB) - Set de evaluación con respuestas cortas
- **Investigathon_LLMTrack_Evaluation_s_cleaned.json** (128.2 MB) - Set de evaluación completo con respuestas
- **Investigathon_LLMTrack_HeldOut_s_cleaned.json** (128.2 MB) - Set de held-out SIN respuestas (para submisión final)

Los archivos se guardarán en:
- `data/longmemeval/` - Datasets originales
- `data/investigathon/` - Datasets de la competencia

### Ollama

Instalar ollama
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

Chequear que esta corriendo

```
sudo systemctl status ollama
```

Bajar nomic-embed-text, que es el modelo de embeddings que vamos a usar

```
ollama pull nomic-embed-text
```

Bajar tambien Gemma3-4B, el modelo que vamos a usar inicialmente
```
ollama pull gemma3:4b
```

### Correr el benchmark

#### Benchmark original de LongMemEval

Para ejecutar el benchmark original:

```sh
uv run main.py
```

o bien:

```sh
python main.py
```

#### Evaluación en el dataset del Investigathon

Para evaluar tu sistema en el **set de evaluación** (que incluye respuestas correctas):

```sh
python run_evaluation.py --dataset_type full --n_samples 250
```

Este script usa la **misma configuración RAG que `main.py`**. 

Parámetros:
- `--dataset_type`: `oracle` (solo sesiones relevantes) o `full` (todas las ~53 sesiones, ~115k tokens) [default: `full`]
- `--n_samples`: número de muestras a procesar [default: 250]

Los resultados se guardarán en `data/results/` con métricas de accuracy.

**Nota:** Para cambiar la configuración (modelo, embedding, etc.), edita la función `load_config_from_main()` en `run_evaluation.py` o modifica directamente `main.py`.

#### Generar predicciones para el Held-Out Set (SUBMISIÓN FINAL)

Para generar las predicciones del **set held-out** (sin respuestas, para submisión):

```sh
python run_held_out.py --n_samples 250
```

Este script usa la **misma configuración RAG que `main.py`** y genera un archivo JSON con las predicciones que debes entregar antes del **11/12 a las 16:00**.

Parámetros:
- `--n_samples`: número de muestras a procesar [default: 250]
- `--output_file`: ruta del archivo de salida (opcional)

El formato de salida será:

```json
[
  {
    "question_id": "...",
    "predicted_answer": "..."
  },
  ...
]
```

Los resultados se guardarán en `data/results/investigathon_heldout_*.json`

**Nota:** Para cambiar la configuración (modelo, embedding, etc.), edita la función `load_config_from_main()` en `run_held_out.py` o modifica directamente `main.py`.

### Analizar resultados

En `notebooks/rag_result_eval.ipynb` encontrarás un análisis general de los resultados, segmentado por tipo de pregunta. Recomendamos reportar las métricas siguiendo esa segmentación, ya que cada categoría presenta distintos niveles de dificultad.

Para correr este notebook con el mismo env, deben hacer primero 

```sh
uv pip install ipykernel
```
