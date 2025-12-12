import json
from pathlib import Path

from src.datasets.LongMemEvalDataset import LongMemEvalDataset

# ⚠️ CAMBIÁ ESTO si tu carpeta de resultados tiene otro nombre.
# Copiá EXACTAMENTE lo que te imprime main.py en:
# "Results will be saved to: ..."
RESULTS_DIR = Path(
    "/home/ubuntu/investigaton-NLP-1/data/results/investigathon_held_out/short/embeddings_gemini_text-embedding-004_memory_ollama_gemma3:4b_judge_ollama_gemma3:4b"
)

OUTPUT_JSONL = Path("submissions/heldout_submission_gemma_gemini.jsonl")
OUTPUT_JSON = Path("submissions/heldout_submission_gemma_gemini.json")


def main() -> None:
    # 1) Cargar el dataset held-out con argumentos POSICIONALES
    dataset = LongMemEvalDataset(
        "short",                  # dataset_type
        "investigathon_held_out", # dataset_set
    )

    print(f"Dataset held-out cargado: {len(dataset)} filas esperadas (deberían ser 250)")

    if not RESULTS_DIR.exists():
        raise SystemExit(f"[ERROR] RESULTS_DIR no existe: {RESULTS_DIR}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing_ids: list[str] = []

    for instance in dataset:
        qid = instance.question_id
        result_path = RESULTS_DIR / f"{qid}.json"

        if not result_path.exists():
            missing_ids.append(qid)
            continue

        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Objeto EXACTO que pide la consigna
        row = {
            "question_id": data.get("question_id", qid),
            "question": data.get("question", instance.question),
            "predicted_answer": data.get("predicted_answer", ""),
        }
        rows.append(row)

    print(f"Total de filas recopiladas: {len(rows)}")

    if missing_ids:
        print(f"[WARN] Faltan {len(missing_ids)} preguntas sin JSON en {RESULTS_DIR}")
        # Si querés ver cuáles son:
        # print(missing_ids)
    else:
        print("✅ Hay resultados para todas las preguntas del held-out.")

    # 2) Guardar en formato JSONL (un JSON por línea)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Archivo JSONL guardado en: {OUTPUT_JSONL}")

    # 3) Opcional: guardar también como lista JSON "grande"
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"✅ Archivo JSON (lista) guardado en: {OUTPUT_JSON}")

    if len(rows) != len(dataset):
        print(
            "[WARN] Ojo: número de filas != número de preguntas en el held-out.\n"
            "       Revisá qué falta antes de enviar."
        )


if __name__ == "__main__":
    main()
