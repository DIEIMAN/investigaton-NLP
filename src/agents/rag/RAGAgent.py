import os
import pickle
import numpy as np
from tqdm import tqdm
from src.agents.MemoryAgent import MemoryAgent
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


instance_embeddings_path = "data/rag/instance_embeddings.pkl"
if os.path.exists(instance_embeddings_path):
    with open(instance_embeddings_path, "rb") as f:
        instance_embeddings = pickle.load(f)
else:
    instance_embeddings = {}
    save_pickle(instance_embeddings, instance_embeddings_path)


def embed_text(message):
    response = embedding(model="azure/text-embedding-3-small", input=message)
    return response.data[0]["embedding"]


def get_rounds_and_embeddings(instance: LongMemEvalInstance):
    if not instance.question_id in instance_embeddings:
        rounds = []
        embeddings = []
        for session in tqdm(instance.sessions, desc="Embedding sessions"):
            for i in range(0, len(session.messages) - 1, 2):
                round = f"{session.messages[i]['role']}: {session.messages[i]['content']}\n{session.messages[i + 1]['role']}: {session.messages[i + 1]['content']}"
                rounds.append(round)
                embeddings.append(embed_text(round))
        instance_embeddings[instance.question_id] = (rounds, embeddings)
        save_pickle(instance_embeddings, instance_embeddings_path)
    return instance_embeddings[instance.question_id]


def retrieve_most_relevant_rounds(instance: LongMemEvalInstance, k: int):

    question_embedding = embed_text(instance.question)
    rounds, embeddings = get_rounds_and_embeddings(instance)

    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_rounds_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_rounds = [rounds[i] for i in most_relevant_rounds_indices]

    return most_relevant_rounds


class RAGAgent(MemoryAgent):
    def __init__(self, model="azure/gpt-4.1"):
        self.model = model

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_rounds = retrieve_most_relevant_rounds(instance, 10)

        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {most_relevant_rounds}
        The question is: {instance.question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
