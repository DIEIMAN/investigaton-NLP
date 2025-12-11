# test_embedding.py
import os
from litellm import embedding

def main():
    model_name = "gemini/text-embedding-004"

    print("=== Test de embeddings con LiteLLM ===")
    print("Modelo:", model_name)

    # Ver variables de entorno relevantes
    gemini_key = os.getenv("GEMINI_API_KEY")
    print("GEMINI_API_KEY presente?:", bool(gemini_key))
    if gemini_key:
        print("Longitud de GEMINI_API_KEY:", len(gemini_key))

    try:
        resp = embedding(
            model=model_name,
            input=["hola, probando embeddings con Gemini"]
        )
        print("\nRespuesta cruda de embedding():")
        print(resp)

        # Intentar leer el primer vector
        emb = resp.data[0]["embedding"]
        print("\nDimensión del embedding:", len(emb))
        print("Primeros 5 valores:", emb[:5])

    except Exception as e:
        print("\n[ERROR] Falló la llamada a embedding():")
        print(repr(e))

if __name__ == "__main__":
    main()
