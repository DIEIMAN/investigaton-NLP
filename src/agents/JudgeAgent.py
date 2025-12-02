from src.datasets.LongMemEvalDataset import LongMemEvalInstance


class JudgeAgent:
    """
    Agente juez: usa un LLM para evaluar si la respuesta del agente de memoria
    (RAGAgent) es correcta o no, comparándola con la ground truth.

    Este agente implementa el esquema de "LLM-as-a-judge".
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model :
            Modelo de lenguaje envuelto (LiteLLMModel u otro) que expone
            un método `reply(messages)` compatible con el formato ChatCompletion.
        """
        self.model = model

    def judge(self, instance: LongMemEvalInstance, predicted_answer):
        """
        Evalúa si `predicted_answer` es correcta para la pregunta dada por
        `instance.question`, usando como referencia `instance.answer`.

        Construye un prompt en inglés que le pide al LLM que devuelva
        exclusivamente `True` o `False`.

        Parameters
        ----------
        instance : LongMemEvalInstance
            Instancia del benchmark con pregunta + respuesta correcta.
        predicted_answer : str
            Respuesta generada por el agente de memoria (RAGAgent).

        Returns
        -------
        bool
            True si el modelo juez considera correcta la predicción,
            False en caso contrario.
        """

        # Prompt simple: el juez ve la pregunta, la predicción y la ground truth.
        # Importante: se le pide explícitamente que devuelva SOLO True o False.
        prompt = f"""
        You are a helpful assistant that judges the correctness of an answer to a question.
        The question is: {instance.question}
        The memory agent answer is: {predicted_answer}
        The ground truth answer is: {instance.answer}
        Return True if the prediction is correct, False otherwise. No other text or explanation.
        """

        # Formato de mensajes tipo ChatCompletion
        messages = [{"role": "user", "content": prompt}]

        # Llamada al modelo juez
        judgment = self.model.reply(messages)

        # El modelo debería devolver exactamente 'True' o 'False'.
        # Se usa eval(judgment) para convertir el string a booleano.
        # (En un entorno duro de producción convendría parsear de forma más segura.)
        return eval(judgment)