from typing import Literal
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration class for LongMemEval experiments.

    Esta clase concentra todos los parámetros configurables del experimento.
    Usar Pydantic nos da validación de tipos y valores, lo que evita errores silenciosos.
    """

    # Nombre del modelo que actúa como "asistente con memoria" (RAG agent).
    memory_model_name: str = Field(
        ...,
        description="Memory model name",
    )

    # Modelo de embeddings a utilizar en el retriever semántico.
    # Por defecto usan nomic-embed-text servido por ollama.
    embedding_model_name: str = Field(
    default="gemini/text-embedding-004",
    description="Name of the embedding model",
)

    # Nombre del modelo que se usa como juez (LLM-as-a-judge).
    judge_model_name: str = Field(
        ...,
        description="Judge model name",
    )

    # Tipo de dataset LongMemEval:
    # - "oracle": solo las sesiones que contienen evidencia relevante
    # - "short": historial completo pero recortado
    # - "long": versión larga (no siempre usada en main.py)
    longmemeval_dataset_type: Literal["oracle", "short", "long"] = Field(
        ...,
        description="Type of LongMemEval dataset to use",
    )

    # Conjunto de datos:
    # - "longmemeval": benchmark original
    # - "investigathon_evaluation": 250 preguntas con respuestas (para evaluar modelos)
    # - "investigathon_held_out": 250 preguntas sin respuestas (para envío final)
    longmemeval_dataset_set: Literal[
        "longmemeval",
        "investigathon_evaluation",
        "investigathon_held_out",
    ] = Field(
        ...,
        description="Set of LongMemEval dataset to use",
    )

    # Cantidad de samples a procesar en esta corrida
    N: int = Field(
        default=10,
        description="Number of samples to process",
    )

    class Config:
        """
        Pydantic internal configuration.

        - extra = "forbid": no permite campos no declarados (evita typos).
        - validate_assignment = True: si se reasigna un atributo, se vuelve a validar.
        """

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment