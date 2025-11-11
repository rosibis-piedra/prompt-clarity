from flask import request, jsonify
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def handler(request):
    if request.method == 'OPTIONS':
        return '', 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
    
    data = request.get_json()
    word = data.get('word', '')
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    try:
        word_emb = get_embedding(word)
        
        contexts = {
            "técnico": ["archivo", "documento", "código", "sistema"],
            "emocional": ["sentimiento", "corazón", "alma", "emoción"],
            "físico": ["objeto", "material", "cuerpo", "cosa"],
            "abstracto": ["idea", "concepto", "pensamiento", "noción"]
        }
        
        scores = {}
        for name, words in contexts.items():
            ctx_embs = [get_embedding(w) for w in words]
            ctx_avg = np.mean(ctx_embs, axis=0)
            scores[name] = float(cosine_similarity(word_emb, ctx_avg))
        
        score_values = list(scores.values())
        max_score = max(score_values)
        clarity = max_score * 100
        ambiguity = 1 - max_score
        
        if ambiguity > 0.7:
            interpretation = "Very ambiguous"
            recommendation = "Use a more specific word in your prompts"
            level = "high"
            emoji = "🚨"
        elif ambiguity > 0.4:
            interpretation = "Moderately ambiguous"
            recommendation = "Consider a clearer alternative"
            level = "medium"
            emoji = "⚡"
        else:
            interpretation = "Clear word"
            recommendation = "Good choice for prompting"
            level = "low"
            emoji = "✅"
        
        result = {
            "word": word,
            "contexts": scores,
            "clarity": {
                "score": float(clarity),
                "ambiguity": float(ambiguity),
                "interpretation": interpretation,
                "recommendation": recommendation,
                "level": level,
                "emoji": emoji
            }
        }
        
        return jsonify(result), 200, {
            'Access-Control-Allow-Origin': '*'
        }
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500