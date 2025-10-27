from dotenv import load_dotenv
from src.experiments.pass_whole_context.longmemeval_experiment import run_experiment


load_dotenv()

if __name__ == "__main__":
    run_experiment(
        config={
            "memory_model_type": "transformers",
            "memory_model_name": "Qwen/Qwen3-4B",
            "memory_model_quantized": True,
            "judge_model_name": "azure/gpt-4.1",
            "longmemeval_dataset_type": "oracle",
            "N": 10,
        }
    )
