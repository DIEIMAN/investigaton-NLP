import json
from pathlib import Path
import statistics as stats

# ⚠️ Ajustar si cambiaste modelos / dataset-set / dataset-type
RESULTS_DIR = Path(
    "data/results/investigathon_evaluation/short/"
    "embeddings_gemini_text-embedding-004_memory_ollama_gemma3:4b_judge_ollama_gemma3:4b"
)

files = sorted(RESULTS_DIR.glob("*.json"))

if not files:
    print(f"No se encontraron archivos JSON en {RESULTS_DIR}")
    raise SystemExit(1)

latencies = []
contexts_chars = []
contexts_msgs = []
correct_flags = []

for path in files:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    latencies.append(float(data.get("latency_seconds", 0.0)))
    contexts_chars.append(int(data.get("context_chars", 0)))
    contexts_msgs.append(int(data.get("context_messages", 0)))

    # Sólo los datasets con ground truth tienen este campo
    if "answer_is_correct" in data:
        correct_flags.append(bool(data["answer_is_correct"]))

n = len(files)
print(f"Total questions (archivos JSON): {n}")

# Accuracy
if correct_flags:
    accuracy = sum(correct_flags) / len(correct_flags)
    print(f"Accuracy: {accuracy:.3f}")
else:
    print("Accuracy: N/A (no hay 'answer_is_correct' en los JSON)")

# Latencia
print(f"Latency avg (s): {stats.mean(latencies):.3f}")
if len(latencies) > 1:
    print(f"Latency variance: {stats.pvariance(latencies):.3f}")

# Contexto (chars)
chars_avg = stats.mean(contexts_chars)
print(f"Context chars avg: {chars_avg:.1f}")
if len(contexts_chars) > 1:
    chars_std = stats.pstdev(contexts_chars)
    chars_var = stats.pvariance(contexts_chars)
    print(f"Context chars std: {chars_std:.1f}")
    print(f"Context chars variance: {chars_var:.1f}")

# Contexto (mensajes)
msgs_avg = stats.mean(contexts_msgs)
print(f"Context messages avg: {msgs_avg:.2f}")
if len(contexts_msgs) > 1:
    msgs_std = stats.pstdev(contexts_msgs)
    msgs_var = stats.pvariance(contexts_msgs)
    print(f"Context messages std: {msgs_std:.2f}")
    print(f"Context messages variance: {msgs_var:.2f}")

