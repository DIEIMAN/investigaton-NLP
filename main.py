from dotenv import load_dotenv
from src.experiments.pass_whole_context.longmemeval_experiment import run_experiment
from config.config import Config, MemoryModelConfig


load_dotenv()

if __name__ == "__main__":
    config = Config(
        memory_model=MemoryModelConfig(
            model_type="litellm",
            model_name="azure/gpt-5-mini",
        ),
        memory_agent="RAG",
        judge_model_name="azure/gpt-5-mini",
        longmemeval_dataset_type="short",
        N=-1,
    )
    run_experiment(config)


# Otros ejemplos de configuracion
# Si pasamos esta MemoryModelConfig, vamos a usar un modelo de transformers de huggingface.
# MemoryModelConfig(
#     model_type="transformers",
#     model_name="Qwen/Qwen3-4B",
# ),
# Para que esto funcione, necesitan correrlo en una maquin acorde.

# Tambien podemos usar un modelo con mucho contexto para probar.
# Otro detalle, para los modelos de openai, se puede especificar el proveedor.
# config = Config(
#     memory_model=MemoryModelConfig(
#         model_type="litellm",
#         model_name="openai/gpt-5",
#     ),
#     memory_agent="RAG",
#     judge_model_name="openai/gpt-5",
#     longmemeval_dataset_type="short",
#     N=-1,
# )
