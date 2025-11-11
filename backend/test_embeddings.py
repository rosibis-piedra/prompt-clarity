from openai import OpenAI
import numpy as np

# PON TU API KEY AQUÍ (la conseguiremos en el próximo paso)
client = OpenAI(api_key="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")

def get_embedding(text):
    """Obtiene embedding de una palabra"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    """Calcula similitud entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# PRUEBA
print("🔬 Obteniendo embedding de 'convertir'...")
embedding = get_embedding("convertir")
print(f"✅ Dimensiones: {len(embedding)}")
print(f"✅ Primeros 5 valores: {embedding[:5]}")

print("\n🔬 Comparando palabras...")
words = ["convertir", "transformar", "cocinar"]
embeddings = {word: get_embedding(word) for word in words}

print("\nSimilitudes:")
base = embeddings["convertir"]
for word in ["transformar", "cocinar"]:
    sim = cosine_similarity(base, embeddings[word])
    print(f"  convertir ↔ {word}: {sim:.3f}")

print("\n✨ ¡Funciona!")