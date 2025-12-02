import argparse
import json
import os
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel #Modelo que responde usando el RAG
from src.agents.JudgeAgent import JudgeAgent #Juez
from src.agents.RAGAgent import RAGAgent #Modelo de memoria + nombre del modelo de embeddings
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config

load_dotenv()


def parse_args():
    """
    Define y parsea los argumentos de línea de comando para configurar el experimento.
    """
    parser = argparse.ArgumentParser(description="Run LongMemEval evaluation pipeline")

    # Modelo que actúa como "asistente con memoria" (el que responde usando RAG)
    parser.add_argument(
        "--memory-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Model name for memory/RAG agent (default: ollama/gemma3:4b)"
    )

    # Modelo que actúa como juez (LLM que evalúa si la respuesta es correcta)
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model name for judge agent (default: openai/gpt-5-mini)"
    )

    # Tipo de dataset: "oracle" (solo sesiones relevantes) o "short" (historial corto).
    # En el paper existe también "long", pero acá por CLI solo usan oracle/short.
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="short",
        choices=["oracle", "short"],
        help="Dataset type: oracle, short (default: short)"
    )

    # Conjunto de datos: benchmark original o los sets del Investigathon.
    parser.add_argument(
        "--dataset-set",
        type=str,
        default="longmemeval",
        choices=["longmemeval", "investigathon_evaluation", "investigathon_held_out"],
        help="Dataset set to use (default: longmemeval)"
    )

    # Cantidad de samples a procesar en esta corrida.
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    return parser.parse_args()

# Parseamos argumentos

args = parse_args()

# Construimos objeto de configuración tipado
config = Config(
    memory_model_name=args.memory_model,
    judge_model_name=args.judge_model,
    longmemeval_dataset_type=args.dataset_type,
    longmemeval_dataset_set=args.dataset_set,
    N=args.num_samples,
)

print(f"\nInitializing models...")
print(f"  Memory Model: {config.memory_model_name}")
print(f"  Judge Model: {config.judge_model_name}")
print(f"  Embedding Model: {config.embedding_model_name}")

memory_model = LiteLLMModel(config.memory_model_name) # Modelo de memoria (RAG) envuelto en LiteLLM

judge_model = LiteLLMModel(config.judge_model_name) # Modelo juez envuelto en LiteLLM
judge_agent = JudgeAgent(model=judge_model) # Agente juez que usa el modelo anterior y un prompt de evaluación

memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name) # Agente RAG que usa el modelo de memoria y un modelo de embeddings

# Cargamos el dataset (LongMemEval + variaciones Investigathon)

longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type, config.longmemeval_dataset_set)

# Create results directory
results_dir = f"data/results/{config.longmemeval_dataset_set}/{config.longmemeval_dataset_type}/embeddings_{config.embedding_model_name.replace('/', '_')}_memory_{config.memory_model_name.replace('/', '_')}_judge_{config.judge_model_name.replace('/', '_')}"
os.makedirs(results_dir, exist_ok=True)

print(f"\nResults will be saved to: {results_dir}")
print(f"Processing samples...")
print("=" * 100)

# Bucle principal: procesar N instancias del dataset
for instance in longmemeval_dataset[: config.N]:
    # Un archivo JSON por pregunta

    result_file = f"{results_dir}/{instance.question_id}.json"
    
    # Si ya existe resultado para esa pregunta, la salteamos
    if os.path.exists(result_file):
        print(f"Skipping {instance.question_id} because it already exists", flush=True)
        continue

    predicted_answer = memory_agent.answer(instance)     # Usamos el agente de memoria (RAG) para generar una respuesta


    if config.longmemeval_dataset_set != "investigathon_held_out":     # Si no es held-out, también evaluamos con el juez

        answer_is_correct = judge_agent.judge(instance, predicted_answer)

       # Guardamos resultado en JSON
    with open(result_file, "w", encoding="utf-8") as f:
        result = {
            "question_id": instance.question_id,
            "question": instance.question,
            "predicted_answer": predicted_answer,
        }
        # En los sets con ground truth guardamos también la respuesta correcta y el flag de correctitud
        if config.longmemeval_dataset_set != "investigathon_held_out":
            result["answer"] = instance.answer
            result["answer_is_correct"] = answer_is_correct

        json.dump(result, f, indent=2)

        # Logging simple a consola
        print(f"  Question: {instance.question}...")
        print(f"  Predicted: {predicted_answer}")
        if config.longmemeval_dataset_set != "investigathon_held_out":
            print(f"  Ground Truth: {instance.answer}")
            print(f"  Correct: {answer_is_correct}")
        print("-" * 100)

print("EVALUATION COMPLETE")