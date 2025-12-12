import json
from pathlib import Path
from collections import Counter  # 👈 nuevo

p = Path("data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json")

with p.open("r", encoding="utf-8") as f:
    rows = json.load(f)

print(f"Nº de filas en el dataset: {len(rows)}\n")

row0 = rows[0]
print("Keys de la primera fila:\n")
for k in row0.keys():
    print(" -", k)

print("\nResumen de cada key en la primera fila:\n")
for k, v in row0.items():
    v_type = type(v).__name__
    if isinstance(v, (str, list, dict)):
        preview = repr(v)
        if len(preview) > 300:
            preview = preview[:297] + "..."
    else:
        preview = repr(v)
    print(f"Key: {k}")
    print(f"  Tipo: {v_type}")
    print(f"  Ejemplo: {preview}\n")


# Lista con todos los question_type
question_types = [row["question_type"] for row in rows]

# Valores distintos
print("Valores únicos de question_type:")
print(set(question_types))

# Cantidad de veces que aparece cada uno
print("\nFrecuencias de question_type:")
print(Counter(question_types))
    