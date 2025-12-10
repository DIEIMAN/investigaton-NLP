import os
import time
import math
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from rerankers import Reranker   # <--- NUEVO
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding  # wrapper unificado de LiteLLM

# --- Configuración de reintentos para embedding ---
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5  # exponencial


def robust_embed_texts(
    texts: List[str],
    embedding_model_name: str,
    api_base: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> List[Optional[List[float]]]:
    """
    Recibe una lista de textos y devuelve una lista de embeddings (o None si falla para un texto).
    Hace la llamada en batch cuando sea posible y reintenta en caso de errores transitorios.
    Parámetros:
      - texts: lista de strings a embedir
      - embedding_model_name: nombre del modelo (ej: "ollama/nomic-embed-text" o "nomic-embed-text")
      - api_base: opcional, por ejemplo "http://localhost:11434"
      - provider_hint: opcional, si LiteLLM reclama el 'provider' (p.ej "ollama")
    """
    # Remover strings vacíos y mantener indices para recomponer
    if not texts:
        return []

    # Intentar en batch; litellm.embedding típicamente acepta lista en input
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            kwargs = {"model": embedding_model_name, "input": texts}
            if api_base:
                kwargs["api_base"] = api_base
            if provider_hint:
                # en algunas versiones litellm requiere una forma de modelo con provider; la dejamos como hint
                kwargs["model"] = embedding_model_name  # keep, user can pass full provider path
            resp = embedding(**kwargs)
            # resp.data es una lista de objetos con "embedding"
            out = []
            for item in resp.data:
                if item is None or "embedding" not in item:
                    out.append(None)
                else:
                    out.append(item["embedding"])
            return out
        except Exception as e:
            attempt += 1
            wait = BACKOFF_FACTOR ** attempt
            print(f"[embed] attempt {attempt} failed: {e}. retrying in {wait:.1f}s...")
            time.sleep(wait)

    # Si falló todo, devolvemos None para cada texto
    print("[embed] all retries failed, returning None embeddings")
    return [None] * len(texts)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    prefer_sentence_boundary: bool = True,
) -> List[str]:
    """
    Divide `text` en chunks de hasta `chunk_size` caracteres con `chunk_overlap` de solapamiento.
    - evita cortar palabras (trata de retroceder hasta un espacio)
    - si prefer_sentence_boundary=True intenta cortar en puntos/fin de oración cuando sea posible
    """
    if not text:
        return []

    # Normalización básica
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)

        # preferir final de oración cercano al end
        if prefer_sentence_boundary and end < n:
            # buscar ., ? o ! desde end-1 hacia start
            segment = text[start:end]
            # buscar el último punto/fin de frase en segment (excluyendo abreviaturas complicadas)
            m = re.search(r'([\.!?])\s+[A-ZÁÉÍÓÚÑ]', text[start:end + 20])  # lookahead simple
            # alternativa más simple: buscar el último .!? dentro del segmento
            last_punct = max(segment.rfind('.'), segment.rfind('!'), segment.rfind('?'))
            if last_punct != -1 and last_punct > int(0.5 * (end - start)):
                end = start + last_punct + 1

        # evitar cortar palabra: retroceder hasta el último espacio si estamos en medio de palabra
        if end < n and text[end].isalnum():
            last_space = text.rfind(" ", start, end)
            if last_space != -1 and last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # mover inicio considerando overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0

        # seguridad: si no avanzamos, forzamos avanzar para evitar loop infinito
        if len(chunks) > 0 and start == (end - chunk_overlap) and end == n:
            break

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

    cache_dir = f"data/rag/embeddings_{embedding_model_name.replace('/', '_')}"
    cache_path = os.path.join(cache_dir, f"{instance.question_id}.parquet")

    # Leer caché si existe
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        # df["embeddings"] debería ser listas de floats o None
        messages = df["messages"].tolist()
        embeddings_raw = df["embeddings"].tolist()
        embeddings = [
            np.array(e, dtype=np.float32) if (e is not None) else None for e in embeddings_raw
        ]
        return messages, embeddings

    # Si no hay caché, construir lista de chunks
    messages = []
    texts_to_embed = []
    # Mantener mapeo de index en texts_to_embed -> index en messages
    mapping = []

    for session_idx, session in enumerate(tqdm(instance.sessions, desc="Chunking sessions")):
        for msg_idx, message in enumerate(session.messages):
            role = message.get("role", "user")
            content = message.get("content", "")
            # chunkear el contenido
            chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

    # Embedir en batches
    embeddings: List[Optional[np.ndarray]] = [None] * len(messages)
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches"):
        batch_texts = texts_to_embed[i : i + batch_size]
        batch_embeddings = robust_embed_texts(batch_texts, embedding_model_name, api_base, provider_hint)
        for j, emb in enumerate(batch_embeddings):
            global_idx = i + j
            msg_idx = mapping[global_idx]
            if emb is None:
                embeddings[msg_idx] = None
            else:
                embeddings[msg_idx] = np.array(emb, dtype=np.float32)

    # Asegurar carpeta y guardar caché (guardamos listas simples para parquet)
    os.makedirs(cache_dir, exist_ok=True)
    # convertir embeddings a listas o None para parquet
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
    Recupera los k mensajes más relevantes usando producto punto entre embeddings (cosine opcional).
    Ignora mensajes cuyo embedding sea None.
    """
    # Embedding de la pregunta
    q_emb_list = robust_embed_texts([instance.question], embedding_model_name, api_base, provider_hint)
    q_emb = q_emb_list[0]
    if q_emb is None:
        raise RuntimeError("No se pudo generar embedding para la pregunta.")

    q_emb = np.array(q_emb, dtype=np.float32)

    messages, embeddings = get_messages_and_embeddings(
        instance,
        embedding_model_name,
        api_base=api_base,
        provider_hint=provider_hint,
    )

    # Filtrar mensajes con embeddings válidos
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    if not valid_indices:
        return []

    mat = np.stack([embeddings[i] for i in valid_indices], axis=0)  # shape (M, D)

    # normalizar para usar cosine similarity (opcionalmente se puede usar dot product directo)
    def safe_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm == 0 or math.isnan(norm):
            return x
        return x / norm

    qn = safe_normalize(q_emb)
    matn = np.array([safe_normalize(row) for row in mat])

    scores = matn @ qn  # cosine similitudes
    # obtener top-k sobre los índices válidos
    topk_idx = np.argsort(scores)[::-1][:k]
    selected_global_indices = [valid_indices[i] for i in topk_idx]
    most_relevant_messages = [messages[i] for i in selected_global_indices]

    return most_relevant_messages


class RAGAgent:
    def __init__(self, model, embedding_model_name: str, api_base: Optional[str] = None, provider_hint: Optional[str] = None):
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.api_base = api_base
        self.provider_hint = provider_hint

        # ⬇️ NUEVO: inicializar el reranker
        self.reranker = Reranker("BAAI/bge-reranker-base")

    def answer(self, instance: LongMemEvalInstance, k: int = 10) -> Tuple[str, Dict]:
        # Paso 1: Recuperación por embeddings
        most_relevant = retrieve_most_relevant_messages(
            instance,
            k,
            self.embedding_model_name,
            api_base=self.api_base,
            provider_hint=self.provider_hint,
        )

        # Paso 2: Aplicar RERANKING sobre esos k documentos
        if most_relevant:
            docs_for_rerank = [{"text": f"{m['role']}: {m['content']}"} for m in most_relevant]
            reranked = self.reranker.rerank(instance.question, docs_for_rerank, top_k=min(5, len(docs_for_rerank)))

            # extraer texto rerankeado y volver al formato original de mensajes
            reranked_texts = [item["text"] for item in reranked]
            # mapear de vuelta a msg original
            new_relevant = []
            text_to_msg = {f"{m['role']}: {m['content']}": m for m in most_relevant}
            for t in reranked_texts:
                if t in text_to_msg:
                    new_relevant.append(text_to_msg[t])

            # reemplazamos
            most_relevant = new_relevant

        # Paso 3: Construcción del contexto final
        context_str = ""
        for msg in most_relevant:
            context_str += f"[{msg['role']}] {msg['content']}\n"

        context_info = {
            "context_messages": len(most_relevant),
            "context_chars": len(context_str),
        }

        # Paso 4: Prompt final al modelo
        prompt = (
            "You are a helpful assistant that answers a question based on the evidence below.\n\n"
            f"Evidence:\n{context_str}\n"
            f"Question: {instance.question}\n\n"
            "Provide a concise, factual answer. If the answer is not in the evidence, say you don't know."
        )

        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)

        return answer, context_info
