import argparse
import json
from pathlib import Path
import statistics as stats
from collections import defaultdict
from typing import TypedDict, DefaultDict, Any

import pandas as pd
from src.datasets.LongMemEvalDataset import LongMemEvalDataset


# ------------------------------
# Tipos para Pylance
# ------------------------------

class TypeStats(TypedDict):
    latencies: list[float]
    context_chars: list[int]
    context_msgs: list[int]
    correct_flags: list[bool]
    question_ids: list[str]


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analizar resultados de LongMemEval / Investigathon"
    )

    parser.add_argument(
        "--dataset-set",
        type=str,
        default="investigathon_evaluation",
        choices=["longmemeval", "investigathon_evaluation", "investigathon_held_out"],
        help="Dataset set: longmemeval, investigathon_evaluation, investigathon_held_out",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        default="short",
        choices=["oracle", "short", "long"],
        help="Dataset type: oracle, short, long (según lo que exista en el repo)",
    )

    parser.add_argument(
        "--memory-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Memory model name (debe matchear el usado en main.py)",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Judge model name (debe matchear el usado en main.py)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gemini/text-embedding-004",
        help="Embedding model name (debe matchear el usado en main.py)",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Tag de corrida usado en main.py (ej: 'baseline_chunk500_k10')",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Ruta explícita al directorio de resultados (sobrescribe el armado automático)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------
    # Armar results_dir
    # ------------------------------
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        tag_suffix = f"_run-{args.run_tag}" if args.run_tag else ""
        results_dir = Path(
            "data/results"
        ) / args.dataset_set / args.dataset_type / (
            "embeddings_"
            + args.embedding_model.replace("/", "_")
            + "_memory_"
            + args.memory_model.replace("/", "_")
            + "_judge_"
            + args.judge_model.replace("/", "_")
            + tag_suffix
        )

    print(f"Usando results_dir = {results_dir}")

    if not results_dir.exists():
        print(f"[ERROR] El directorio {results_dir} no existe.")
        raise SystemExit(1)

    # ------------------------------
    # Cargar dataset base con LongMemEvalDataset
    # ------------------------------

    try:
        lm_dataset = LongMemEvalDataset(args.dataset_type, args.dataset_set)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar LongMemEvalDataset({args.dataset_type}, {args.dataset_set}): {e}")
        raise

    dataset_df: pd.DataFrame = lm_dataset.dataset

    if "question_id" not in dataset_df.columns or "question_type" not in dataset_df.columns:
        print("[ERROR] El dataset no tiene columnas 'question_id' y 'question_type'.")
        print(f"Columnas disponibles: {list(dataset_df.columns)}")
        raise SystemExit(1)

    qtype_by_qid: dict[str, str] = {}
    for _, row in dataset_df[["question_id", "question_type"]].iterrows():
        qid = str(row["question_id"])
        qtype = str(row["question_type"])
        qtype_by_qid[qid] = qtype

    print(f"Cargado dataset base: {len(dataset_df)} filas")
    print(f"Question IDs únicos en dataset: {len(qtype_by_qid)}\n")

    # ------------------------------
    # Cargar resultados
    # ------------------------------

    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"No se encontraron archivos JSON en {results_dir}")
        raise SystemExit(1)

    print(f"Usando {len(files)} archivos de resultados\n")

    latencies: list[float] = []
    contexts_chars: list[int] = []
    contexts_msgs: list[int] = []
    correct_flags: list[bool] = []

    per_type: DefaultDict[str, TypeStats] = defaultdict(
        lambda: {
            "latencies": [],
            "context_chars": [],
            "context_msgs": [],
            "correct_flags": [],
            "question_ids": [],
        }
    )

    missing_qid: int = 0
    unknown_type: int = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        # --- métricas globales ---
        latency = float(data.get("latency_seconds", 0.0))
        ctx_chars = int(data.get("context_chars", 0))
        ctx_msgs = int(data.get("context_messages", 0))

        latencies.append(latency)
        contexts_chars.append(ctx_chars)
        contexts_msgs.append(ctx_msgs)

        is_correct: bool | None = None
        if "answer_is_correct" in data:
            is_correct = bool(data["answer_is_correct"])
            correct_flags.append(is_correct)

        # --- agrupar por question_type ---
        raw_qid = data.get("question_id")
        if raw_qid is None:
            missing_qid += 1
            continue

        qid = str(raw_qid)

        if qid in qtype_by_qid:
            qtype: str = qtype_by_qid[qid]
        else:
            qtype = "UNKNOWN"
            unknown_type += 1

        bucket: TypeStats = per_type[qtype]
        bucket["latencies"].append(latency)
        bucket["context_chars"].append(ctx_chars)
        bucket["context_msgs"].append(ctx_msgs)
        bucket["question_ids"].append(qid)
        if is_correct is not None:
            bucket["correct_flags"].append(is_correct)

    # ------------------------------
    # Preparar filas para CSV
    # ------------------------------
    rows_for_csv: list[dict[str, Any]] = []

    # ------------------------------
    # Métricas globales
    # ------------------------------

    n = len(files)
    print(f"Total questions (archivos JSON): {n}")

    if missing_qid > 0:
        print(f"[WARN] {missing_qid} JSON(s) sin 'question_id' (sólo cuentan a nivel global)")

    if unknown_type > 0:
        print(
            f"[WARN] {unknown_type} JSON(s) con 'question_id' que no aparece en el dataset "
            f"(tipo = 'UNKNOWN')"
        )

    if correct_flags:
        accuracy = sum(correct_flags) / len(correct_flags)
        print(f"Accuracy global: {accuracy:.3f}")
    else:
        accuracy = None
        print("Accuracy global: N/A (no hay 'answer_is_correct' en los JSON)")

    latency_mean = stats.mean(latencies)
    latency_var = stats.pvariance(latencies) if len(latencies) > 1 else 0.0
    print(f"Latency avg (s): {latency_mean:.3f}")
    if len(latencies) > 1:
        print(f"Latency variance: {latency_var:.3f}")

    chars_avg = stats.mean(contexts_chars)
    if len(contexts_chars) > 1:
        chars_std = stats.pstdev(contexts_chars)
        chars_var = stats.pvariance(contexts_chars)
    else:
        chars_std = 0.0
        chars_var = 0.0

    print(f"Context chars avg: {chars_avg:.1f}")
    if len(contexts_chars) > 1:
        print(f"Context chars std: {chars_std:.1f}")
        print(f"Context chars variance: {chars_var:.1f}")

    msgs_avg = stats.mean(contexts_msgs)
    if len(contexts_msgs) > 1:
        msgs_std = stats.pstdev(contexts_msgs)
        msgs_var = stats.pvariance(contexts_msgs)
    else:
        msgs_std = 0.0
        msgs_var = 0.0

    print(f"Context messages avg: {msgs_avg:.2f}")
    if len(contexts_msgs) > 1:
        print(f"Context messages std: {msgs_std:.2f}")
        print(f"Context messages variance: {msgs_var:.2f}")

    # Fila global para CSV
    rows_for_csv.append(
        {
            "scope": "global",
            "question_type": "ALL",
            "num_questions": n,
            "accuracy": accuracy,
            "latency_mean": latency_mean,
            "latency_var": latency_var,
            "context_chars_mean": chars_avg,
            "context_chars_std": chars_std,
            "context_chars_var": chars_var,
            "context_msgs_mean": msgs_avg,
            "context_msgs_std": msgs_std,
            "context_msgs_var": msgs_var,
        }
    )

    print("\nResumen de tipos en esta corrida:")
    for qtype, g in per_type.items():
        print(f"  {qtype}: {len(g['question_ids'])} preguntas")

    # ------------------------------
    # Métricas por question_type
    # ------------------------------

    print("\n==============================")
    print("MÉTRICAS POR question_type")
    print("==============================")

    for qtype, g in per_type.items():
        n_q = len(g["latencies"])
        print(f"\n> {qtype} (n={n_q})")

        if n_q == 0:
            print("  [SIN DATOS]")
            continue

        # Accuracy
        if g["correct_flags"]:
            acc = sum(g["correct_flags"]) / len(g["correct_flags"])
            print(f"  Accuracy: {acc:.3f}")
        else:
            acc = None
            print("  Accuracy: N/A (sin 'answer_is_correct')")

        # Latencia
        lat_mean = stats.mean(g["latencies"])
        lat_var = stats.pvariance(g["latencies"]) if len(g["latencies"]) > 1 else 0.0
        print(f"  Latency avg (s): {lat_mean:.3f}")
        if len(g["latencies"]) > 1:
            print(f"  Latency variance: {lat_var:.3f}")

        # Contexto chars
        c_mean = stats.mean(g["context_chars"])
        if len(g["context_chars"]) > 1:
            c_std = stats.pstdev(g["context_chars"])
            c_var = stats.pvariance(g["context_chars"])
        else:
            c_std = 0.0
            c_var = 0.0
        print(f"  Context chars avg: {c_mean:.1f}")
        if len(g["context_chars"]) > 1:
            print(f"  Context chars std: {c_std:.1f}")
            print(f"  Context chars variance: {c_var:.1f}")

        # Contexto mensajes
        m_mean = stats.mean(g["context_msgs"])
        if len(g["context_msgs"]) > 1:
            m_std = stats.pstdev(g["context_msgs"])
            m_var = stats.pvariance(g["context_msgs"])
        else:
            m_std = 0.0
            m_var = 0.0
        print(f"  Context messages avg: {m_mean:.2f}")
        if len(g["context_msgs"]) > 1:
            print(f"  Context messages std: {m_std:.2f}")
            print(f"  Context messages variance: {m_var:.2f}")

        # Fila por tipo para CSV
        rows_for_csv.append(
            {
                "scope": "per_type",
                "question_type": qtype,
                "num_questions": n_q,
                "accuracy": acc,
                "latency_mean": lat_mean,
                "latency_var": lat_var,
                "context_chars_mean": c_mean,
                "context_chars_std": c_std,
                "context_chars_var": c_var,
                "context_msgs_mean": m_mean,
                "context_msgs_std": m_std,
                "context_msgs_var": m_var,
            }
        )

    # ------------------------------
    # Guardar CSV
    # ------------------------------
    csv_path = results_dir / "metrics_summary.csv"
    df_csv = pd.DataFrame(rows_for_csv)
    df_csv.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nMétricas también guardadas en: {csv_path}")


if __name__ == "__main__":
    main()
