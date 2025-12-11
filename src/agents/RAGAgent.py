import os
import time
import math
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding  # wrapper unificado de LiteLLM

# --- Configuración de reintentos para embedding ---
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5  # exponencial

# --- Límite de tiempo para el chunking+embeddings por instancia ---
MAX_CHUNKING_SECONDS = 60.0


def robust_embed_texts(
    texts: List[str],
    embedding_model_name: str,
    api_base: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> List[Optional[List[float]]]:
    """
    Devuelve una lista de embeddings (o None) para cada texto.
    Soporta tanto respuestas de LiteLLM en forma de objetos Embedding
    como en forma de diccionarios.
    """
    if not texts:
        return []

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            kwargs = {"model": embedding_model_name, "input": texts}
            if api_base:
                kwargs["api_base"] = api_base
            # provider_hint hoy no lo usamos, pero lo dejamos por si acaso
            resp = embedding(**kwargs)

            out: list[Optional[List[float]]] = []
            for item in resp.data:
                if item is None:
                    out.append(None)
                    continue

                vec = None

                # Caso 1: objeto tipo Embedding, con atributo .embedding
                if hasattr(item, "embedding"):
                    vec = item.embedding
                # Caso 2: diccionario {"embedding": [...]}
                elif isinstance(item, dict):
                    vec = item.get("embedding")
                else:
                    # fallback super defensivo
                    try:
                        vec = item["embedding"]  # type: ignore[index]
                    except Exception:
                        vec = None

                if vec is None:
                    out.append(None)
                else:
                    # aseguramos que sea una lista de floats normal
                    out.append(list(vec))

            return out

        except Exception as e:
            attempt += 1
            wait = BACKOFF_FACTOR ** attempt
            print(f"[embed] attempt {attempt} failed: {e}. retrying in {wait:.1f}s...")
            time.sleep(wait)

    print("[embed] all retries failed, returning None embeddings")
    return [None] * len(texts)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    prefer_sentence_boundary: bool = True,  # ya no lo usamos pero queda en la firma
) -> List[str]:
    """
    Versión *segura* de chunk_text:
    - Divide el texto en ventanas de tamaño fijo `chunk_size`
    - Con solapamiento de `chunk_overlap`
    - Sin lógica rara de oraciones ni búsquedas hacia atrás que puedan colgarse
    """
    if not text:
        return []

    # Normalización básica
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n <= chunk_size:
        return [text]

    chunks: List[str] = []

    # Paso fijo: cuánto avanzamos cada vez
    step = max(1, chunk_size - chunk_overlap)

    start = 0
    safety_iters = 0
    MAX_ITERS = 100000  # por seguridad extra

    while start < n and safety_iters < MAX_ITERS:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step
        safety_iters += 1

    if safety_iters >= MAX_ITERS:
        print("[WARN] chunk_text: reached MAX_ITERS, cutting early")

    return chunks


def get_messages_and_embeddings(
    instance: LongMemEvalInstance,
    embedding_model_name: str,
    api_base: Optional[str] = None,
    provider_hint: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 64,
) -> Tuple[List[dict], List[Optional[np.ndarray]]]:
    """
    Obtiene (y cachea) los mensajes chunked y sus embeddings (como arrays numpy o None).
    Retorna:
      - messages: lista de dicts con keys: role, content, original_index, chunk_id
      - embeddings: lista paralela de np.ndarray (float32) o None si falló embedding
    """
    start_time = time.time()

    cache_dir = f"data/rag/embeddings_{embedding_model_name.replace('/', '_')}"
    cache_path = os.path.join(cache_dir, f"{instance.question_id}.parquet")

    # Leer caché si existe
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        messages = df["messages"].tolist()
        embeddings_raw = df["embeddings"].tolist()
        embeddings = [
            np.array(e, dtype=np.float32) if (e is not None) else None
            for e in embeddings_raw
        ]
        return messages, embeddings

    # Si no hay caché, construir lista de chunks
    messages: List[dict] = []
    texts_to_embed: List[str] = []
    mapping: List[int] = []

    for session_idx, session in enumerate(tqdm(instance.sessions, desc="Chunking sessions")):
        # Timeout de chunking
        if time.time() - start_time > MAX_CHUNKING_SECONDS:
            print(f"[WARN] Chunking timeout para question_id={instance.question_id}, cortando early")
            break

        for msg_idx, message in enumerate(session.messages):
            role = message.get("role", "user")
            content = message.get("content", "")

            chunks = chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for cix, chunk in enumerate(chunks):
                msg_record = {
                    "role": role,
                    "content": chunk,
                    "original_session": session_idx,
                    "original_index": msg_idx,
                    "chunk_id": cix,
                }
                messages.append(msg_record)
                texts_to_embed.append(f"{role}: {chunk}")
                mapping.append(len(messages) - 1)

    # Si no se generó ningún mensaje, devolvemos vacío
    if not messages:
        print(f"[WARN] No hay mensajes chunked para question_id={instance.question_id}")
        return [], []

    # Embedir en batches
    embeddings: List[Optional[np.ndarray]] = [None] * len(messages)
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches"):
        # Timeout también en la fase de embeddings
        if time.time() - start_time > MAX_CHUNKING_SECONDS:
            print(f"[WARN] Embedding timeout para question_id={instance.question_id}, cortando early")
            break

        batch_texts = texts_to_embed[i : i + batch_size]
        batch_embeddings = robust_embed_texts(
            batch_texts,
            embedding_model_name,
            api_base,
            provider_hint,
        )
        for j, emb in enumerate(batch_embeddings):
            global_idx = i + j
            if global_idx >= len(mapping):
                break
            msg_idx = mapping[global_idx]
            embeddings[msg_idx] = None if emb is None else np.array(emb, dtype=np.float32)

    # Guardar caché
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_for_disk = [e.tolist() if e is not None else None for e in embeddings]
    df = pd.DataFrame({"messages": messages, "embeddings": embeddings_for_disk})
    df.to_parquet(cache_path, index=False)

    return messages, embeddings


def retrieve_most_relevant_messages(
    instance: LongMemEvalInstance,
    k: int,
    embedding_model_name: str,
    api_base: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> List[dict]:
    """
    Recupera los k mensajes más relevantes usando similitud de coseno entre embeddings.
    Ignora mensajes cuyo embedding sea None.
    """
    # --- Embedding de la pregunta ---
    q_emb_list = robust_embed_texts(
        [instance.question],
        embedding_model_name,
        api_base,
        provider_hint,
    )
    q_emb = q_emb_list[0]
    if q_emb is None:
        raise RuntimeError("No se pudo generar embedding para la pregunta.")

    q_emb = np.array(q_emb, dtype=np.float32)

    # --- Embeddings de los mensajes del historial ---
    messages, embeddings = get_messages_and_embeddings(
        instance,
        embedding_model_name,
        api_base=api_base,
        provider_hint=provider_hint,
    )

    # Si no hay mensajes, devolvemos contexto vacío
    if not messages:
        return []

    # Filtrar mensajes con embeddings válidos
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    if not valid_indices:
        return []

    mat = np.stack([embeddings[i] for i in valid_indices], axis=0)  # shape (M, D)

    # --- Normalizar para usar cosine similarity ---
    def safe_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm == 0 or math.isnan(norm):
            return x
        return x / norm

    qn = safe_normalize(q_emb)
    matn = np.array([safe_normalize(row) for row in mat])

    # Cosine similarities
    scores = matn @ qn

    # Top-k sobre los índices válidos
    topk_idx = np.argsort(scores)[::-1][:k]
    selected_global_indices = [valid_indices[i] for i in topk_idx]
    most_relevant_messages = [messages[i] for i in selected_global_indices]

    return most_relevant_messages


class RAGAgent:
    def __init__(
        self,
        model,
        embedding_model_name: str,
        api_base: Optional[str] = None,
        provider_hint: Optional[str] = None,
    ):
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.api_base = api_base
        self.provider_hint = provider_hint

    def answer(self, instance: LongMemEvalInstance, k: int = 10) -> Tuple[str, Dict]:
        most_relevant = retrieve_most_relevant_messages(
            instance,
            k,
            self.embedding_model_name,
            api_base=self.api_base,
            provider_hint=self.provider_hint,
        )

        if most_relevant:
            context_str = ""
            for msg in most_relevant:
                context_str += f"[{msg['role']}] {msg['content']}\n"
        else:
            context_str = "(No relevant past context found for this question.)\n"

        context_info = {
            "context_messages": len(most_relevant),
            "context_chars": len(context_str),
        }

        prompt = (
            "You are a helpful assistant that answers a question based on the evidence below.\n\n"
            f"Evidence:\n{context_str}\n"
            f"Question: {instance.question}\n\n"
            "Provide a concise, factual answer. If the answer is not in the evidence, say you don't know."
        )

        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)

        return answer, context_info
