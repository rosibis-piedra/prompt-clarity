from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import numpy as np
import os
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# CORS configuration
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "X-CSRF-Token"],
     methods=["GET", "POST", "OPTIONS"])

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day"]
)

# OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# reCAPTCHA secret key (usa la test key por ahora, luego cámbiala por la tuya real)
RECAPTCHA_SECRET_KEY = os.getenv('RECAPTCHA_SECRET_KEY', '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe')

def verify_recaptcha(recaptcha_response):
    """Verify reCAPTCHA response with Google"""
    if not recaptcha_response:
        return False
    
    try:
        verification = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': recaptcha_response
            },
            timeout=5
        )
        result = verification.json()
        return result.get('success', False)
    except Exception as e:
        print(f"reCAPTCHA verification error: {str(e)}")
        return False

def verify_csrf_token(token):
    """Basic CSRF token validation"""
    # En producción, deberías almacenar y validar tokens en sesión/base de datos
    # Por ahora, solo verificamos que existe y tiene el formato correcto
    if not token:
        return False
    # Token debe ser hexadecimal de 64 caracteres (256 bits)
    return len(token) == 64 and all(c in '0123456789abcdef' for c in token.lower())

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/')
def home():
    return jsonify({"status": "Prompt Clarity Backend Running"})

@app.route('/analyze', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    word = data.get('word', '')
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    # SECURITY #2: Verify reCAPTCHA
    recaptcha_response = data.get('g-recaptcha-response', '')
    if not verify_recaptcha(recaptcha_response):
        return jsonify({"error": "reCAPTCHA verification failed"}), 403
    
    # SECURITY #3: Verify CSRF token
    csrf_token = request.headers.get('X-CSRF-Token', '')
    if not verify_csrf_token(csrf_token):
        return jsonify({"error": "Invalid CSRF token"}), 403
    
    try:
        print(f"Analyzing: {word}")
        
        # Análisis
        word_emb = get_embedding(word)
        
        # Contextos
        contexts = {
            "technical": ["file", "document", "code", "system"],
            "emotional": ["feeling", "heart", "soul", "emotion"],
            "physical": ["object", "material", "body", "thing"],
            "abstract": ["idea", "concept", "thought", "notion"]
        }
        
        scores = {}
        for name, words in contexts.items():
            ctx_embs = [get_embedding(w) for w in words]
            ctx_avg = np.mean(ctx_embs, axis=0)
            scores[name] = float(cosine_similarity(word_emb, ctx_avg))
        
        # Calcular ambigüedad y clarity
        score_values = list(scores.values())
        max_score = max(score_values)
        clarity = max_score * 100
        ambiguity = 1 - max_score
        
        # Interpretación
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
        
        print(f"Clarity: {clarity:.1f}% - {interpretation}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Para Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
