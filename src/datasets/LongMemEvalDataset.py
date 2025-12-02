import json
import pandas as pd
import os


class Session:
    """
    Representa una sesión de conversación dentro del historial del usuario.

    Cada instancia tiene:
    - session_id: identificador único de la sesión.
    - date: fecha asociada a la sesión (string o formato que venga del dataset).
    - messages: lista de mensajes dentro de esa sesión.
      Normalmente cada mensaje es un dict tipo:
      {"role": "user" | "assistant", "content": "texto del mensaje"}.
    """

    def __init__(self, session_id, date, messages):
        self.session_id = session_id
        self.date = date
        self.messages = messages

    def __repr__(self):
        return (
            f"Session(session_id={self.session_id}, "
            f"date={self.date}, messages={self.messages})"
        )


class LongMemEvalInstance:
    """
    Representa una instancia completa del benchmark LongMemEval
    (es decir, un ejemplo de evaluación).

    Contiene:
    - question_id: identificador único de la pregunta.
    - question: la pregunta final que el modelo debe responder.
    - sessions: lista de objetos Session que conforman el historial del usuario.
    - t_question: timestamp en el que se hace la pregunta.
    - answer: respuesta correcta (ground truth). En el held-out puede ser None.
    """

    def __init__(self, question_id, question, sessions, t_question, answer):
        self.question_id = question_id
        self.question = question
        self.sessions = sessions
        self.t_question = t_question
        self.answer = answer

    def __repr__(self):
        return (
            "LongMemEvalInstance("
            f"question={self.question}, "
            f"sessions={self.sessions}, "
            f"t_question={self.t_question}, "
            f"answer={self.answer})"
        )


class LongMemEvalDataset:
    """
    Cargador del dataset LongMemEval + extensiones del Investigathon.

    Se comporta de forma similar a un dataset de PyTorch:
    - __len__ devuelve el número total de ejemplos.
    - __getitem__ permite indexar o hacer slicing.
    """

    def __init__(self, type, set):
        """
        Parameters
        ----------
        type : str
            Tipo de dataset:
            - "oracle": solo sesiones relevantes para la pregunta.
            - "short": historial completo pero acotado (~115k tokens máx.).
        set : str
            Conjunto de datos:
            - "longmemeval": benchmark original.
            - "investigathon_evaluation": 250 preguntas con respuestas.
            - "investigathon_held_out": 250 preguntas sin respuestas (para envío).
        """

        # Validación del tipo de dataset
        if type not in ["oracle", "short"]:
            raise ValueError(
                f"Invalid dataset type: {type}. "
                "Must be 'oracle' or 'short'"
            )

        # Validación del conjunto
        if set not in [
            "longmemeval",
            "investigathon_evaluation",
            "investigathon_held_out",
        ]:
            raise ValueError(
                "Invalid dataset set: {set}. Must be "
                "'longmemeval' or 'investigathon_evaluation' or "
                "'investigathon_held_out'"
            )

        # Selección de la ruta del archivo JSON según `set` y `type`.
        if set == "longmemeval":
            path = {
                "oracle": "data/longmemeval/longmemeval_oracle.json",
                "short": "data/longmemeval/longmemeval_s_cleaned.json",
            }[type]

        elif set == "investigathon_evaluation":
            # Evaluación oficial del Investigathon:
            # - oracle: solo sesiones relevantes
            # - short: historial completo (~115k tokens)
            path = {
                "oracle": "data/investigathon/Investigathon_LLMTrack_Evaluation_oracle.json",
                "short": "data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json",
            }[type]

        elif set == "investigathon_held_out":
            # Held-out SOLO tiene versión "short"
            if type != "short":
                raise ValueError(
                    f"Invalid dataset type: {type} for held-out set. "
                    "Must be 'short'"
                )
            path = (
                "data/investigathon/"
                "Investigathon_LLMTrack_HeldOut_s_cleaned.json"
            )

        else:
            # En teoría nunca se llega acá por las validaciones anteriores.
            raise ValueError(
                f"Invalid dataset set: {set}. Must be 'longmemeval' or "
                "'investigathon_evaluation' or 'investigathon_held_out'"
            )

        # Cargamos el JSON y lo convertimos en un DataFrame de pandas.
        # Luego hacemos un shuffle con random_state fijo (42) para obtener
        # siempre el mismo orden (reproducibilidad).
        with open(path, "r", encoding="utf-8") as f:
            self.dataset = (
                pd.DataFrame(json.load(f))
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

        # Puntero interno (no se usa en este código, pero podría servir para
        # iteración secuencial).
        self.current_index = 0

    def __len__(self):
        """
        Devuelve la cantidad total de ejemplos en el dataset.
        """
        return len(self.dataset)

    def __getitem__(self, key):
        """
        Permite indexar el dataset de dos formas:

        - dataset[i]        -> devuelve una instancia LongMemEvalInstance
        - dataset[i:j]      -> devuelve una lista de instancias

        Donde i, j son índices de fila del DataFrame interno.
        """
        sliced_data = self.dataset.iloc[key]

        if isinstance(key, slice):
            # Si es un slice, iteramos filas y construimos una lista de instancias
            return [
                self.instance_from_row(row)
                for _, row in sliced_data.iterrows()
            ]
        else:
            # Si es un índice simple, `sliced_data` ya es una Serie (una fila).
            # Creamos una sola instancia.
            return self.instance_from_row(sliced_data)

    def instance_from_row(self, row):
        """
        Convierte una fila del DataFrame (Serie de pandas) en un
        LongMemEvalInstance bien tipado.

        La fila contiene:
        - question_id
        - question
        - haystack_session_ids
        - haystack_dates
        - haystack_sessions (lista de mensajes por sesión)
        - question_date
        - answer (no presente en el held-out)
        """
        return LongMemEvalInstance(
            question_id=row["question_id"],
            question=row["question"],
            sessions=[
                Session(
                    session_id=session_id,
                    date=date,
                    messages=messages,
                )
                for session_id, date, messages in zip(
                    row["haystack_session_ids"],
                    row["haystack_dates"],
                    row["haystack_sessions"],
                )
            ],
            t_question=row["question_date"],
            # En el held-out, 'answer' no existe, por eso usamos .get()
            answer=row.get("answer"),
        )