import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Tipo LongMemEvalInstance describe una instancia del benchmark
from src.datasets.LongMemEvalDataset import LongMemEvalInstance

# Función de embeddings unificada de LiteLLM
from litellm import embedding


def embed_text(message, embedding_model_name):
    """
    Dado un texto `message`, devuelve su embedding usando el modelo especificado.

    Usa litellm.embedding para abstraer la API del proveedor (ollama, openai, etc.).
    """
    response = embedding(model=embedding_model_name, input=message)
    # `response.data[0]["embedding"]` suele ser una lista de floats
    return response.data[0]["embedding"]


def get_messages_and_embeddings(instance: LongMemEvalInstance, embedding_model_name):
    """
    Para una instancia del benchmark, obtiene todos los mensajes del historial
    (todas las sesiones) y sus embeddings correspondientes.

    Implementa una caché en disco: si ya existe un archivo parquet con los
    mensajes/embeddings para ese `question_id`, lo reutiliza.
    """
    # Ruta de caché específica del modelo de embedding y de la pregunta
    cache_path = (
        f"data/rag/"
        f"embeddings_{embedding_model_name.replace('/', '_')}/"
        f"{instance.question_id}.parquet"
    )

    # Si ya calculamos embeddings para esta pregunta + modelo, leemos el parquet.
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist()

    # Si no existe caché, calculamos todo desde cero.
    messages = []
    embeddings = []

    # Recorremos todas las sesiones y todos los mensajes
    for session in tqdm(instance.sessions, desc="Embedding sessions"):
        for message in session.messages:
            # Guardamos el mensaje tal cual viene (dict con role + content)
            messages.append(message)
            # Embedding del texto "role: content", para incorporar el rol en el vector
            embeddings.append(
                embed_text(
                    f"{message['role']}: {message['content']}",
                    embedding_model_name,
                )
            )

    # Creamos carpetas necesarias para guardar el parquet
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Guardamos mensajes y embeddings como DataFrame para acelerar corridas futuras
    pd.DataFrame({"messages": messages, "embeddings": embeddings}).to_parquet(
        cache_path
    )

    return messages, embeddings


def retrieve_most_relevant_messages(
    instance: LongMemEvalInstance,
    k: int,
    embedding_model_name: str,
):
    """
    Dado una instancia y un entero k:
    - embebe la pregunta,
    - calcula similitud (producto punto) entre la pregunta y cada mensaje del historial,
    - devuelve los k mensajes más similares.

    Este es el "retriever" del baseline: RAG a nivel de mensaje individual.
    """
    # Embedding de la pregunta final
    question_embedding = embed_text(instance.question, embedding_model_name)

    # Obtenemos todos los mensajes del historial y sus embeddings (con caché)
    messages, embeddings = get_messages_and_embeddings(instance, embedding_model_name)

    # Convertimos a matriz/vector y calculamos similitud por producto punto
    similarity_scores = np.dot(embeddings, question_embedding)

    # Ordenamos los índices por similitud descendente
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]

    # Seleccionamos los mensajes más relevantes
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]

    return most_relevant_messages


class RAGAgent:
    """
    Agente de RAG muy simple:

    1. Usa embeddings para recuperar los k mensajes más similares a la pregunta.
    2. Construye un prompt con esos mensajes como "evidence".
    3. Llama al modelo de lenguaje para generar la respuesta.
    """

    def __init__(self, model, embedding_model_name: str):
        # `model` es un LiteLLMModel (o algo compatible) con método `reply`
        self.model = model
        self.embedding_model_name = embedding_model_name

    def answer(self, instance: LongMemEvalInstance):
        """
        Responde la pregunta de una instancia usando RAG baseline.
        """

        # Recuperamos los 10 mensajes más relevantes del historial.
        # Nota: k=10 está hardcodeado; esto es un parámetro obvio para tunear/mejorar.
        most_relevant_messages = retrieve_most_relevant_messages(
            instance,
            10,
            self.embedding_model_name,
        )

        # Prompt muy básico: incluye la evidencia como lista de mensajes en bruto.
        # Podría mejorarse formateando como diálogo legible, agregando instrucciones
        # de no alucinar, razonamiento temporal, etc.
        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {most_relevant_messages}
        The question is: {instance.question}
        Return the answer to the question.
        """

        # Formato de mensajes estilo ChatCompletion
        messages = [{"role": "user", "content": prompt}]

        # Llamada al modelo de lenguaje unificado por LiteLLMModel
        answer = self.model.reply(messages)

        return answer