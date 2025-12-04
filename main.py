"""
main.py - FastAPI backend for Meal Planner AI
Requires Two-Tower and Transformer models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn
import os
import sys

# =========================================================
#                  FASTAPI CONFIGURATION
# =========================================================
# Defines API metadata that appears in /docs
app = FastAPI(
    title="Meal Planner API",
    version="3.0.0",
    description="Recommendation API using Two-Tower and Transformer models (required)."
)

# Enables Cross-Origin access (front-end calling API from localhost or another domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
#                     FILE PATHS
# =========================================================
DATA_DIR = "./data"
MODELS_DIR = "./models"

# Required datasets
REQUIRED_DATA = [
    f"{DATA_DIR}/recipes_clean.csv",
    f"{DATA_DIR}/all_embeddings.csv",
    f"{DATA_DIR}/meals_category.csv"
]

# Required ML models
REQUIRED_MODELS = [
    f"{MODELS_DIR}/trained_user_towers.keras",
    f"{MODELS_DIR}/trained_recipe_towers.keras",
    f"{MODELS_DIR}/TRANSFORMER_MODEL_PRETRAIN.keras"
]

# =========================================================
#              CHECK REQUIRED FILES AT STARTUP
# =========================================================
print("🔄 Checking required files...")

missing_files = []
for f in REQUIRED_DATA + REQUIRED_MODELS:
    if not os.path.exists(f):
        missing_files.append(f)
        print(f"❌ Missing: {f}")

if missing_files:
    print("\n⛔ ERROR: Required files are missing!")
    print("The API cannot start without these resources.")
    sys.exit(1)

# =========================================================
#                   LOAD DATASETS
# =========================================================
print("🔄 Loading datasets...")

# Recipes dataframe
recipes_df = pd.read_csv(f"{DATA_DIR}/recipes_clean.csv", low_memory=False)
if 'id' in recipes_df.columns:
    recipes_df = recipes_df.set_index('id')
print(f"✅ Loaded {len(recipes_df)} recipes.")

# Embeddings (recipe + token vectors)
embeddings_df = pd.read_csv(f"{DATA_DIR}/all_embeddings.csv", index_col=0, low_memory=False)
print(f"✅ Embeddings loaded: {embeddings_df.shape}")

# Meal category mapping
category_df = pd.read_csv(f"{DATA_DIR}/meals_category.csv", low_memory=False)
if 'id' in category_df.columns:
    category_df = category_df.set_index('id')
print("✅ Categories loaded.")

# =========================================================
#                   LOAD MODELS (CRITICAL)
# =========================================================
print("🔄 Loading ML models...")

# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom layers required for model deserialization
from utils.custom_layers import CUSTOM_OBJECTS

def load_keras_model(path):
    """
    Load a Keras model including custom layers.
    Falls back to TensorFlow loader if needed.
    """
    try:
        import keras
        return keras.saving.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    except Exception:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)

user_tower = load_keras_model(f"{MODELS_DIR}/trained_user_towers.keras")
print("✅ User Tower loaded.")

recipe_tower = load_keras_model(f"{MODELS_DIR}/trained_recipe_towers.keras")
print("✅ Recipe Tower loaded.")

transformer_model = load_keras_model(f"{MODELS_DIR}/TRANSFORMER_MODEL_PRETRAIN.keras")
print("✅ Transformer loaded.")

print("✅ Initialization complete — API ready.")

# =========================================================
#                   PYDANTIC REQUEST MODELS
# =========================================================
# These define the expected structure of incoming API requests.

class UserProfile(BaseModel):
    gender: str
    age: int
    height: int
    weight: int
    activity: str
    sport_hours: float = 0
    objective: str = "maintenance"

class RecommendationRequest(BaseModel):
    liked_recipe_ids: List[int]
    disliked_recipe_ids: Optional[List[int]] = []
    n_recommendations: int = 10
    max_calories: Optional[int] = None
    max_time: Optional[int] = None

class PlanningRequest(BaseModel):
    liked_recipe_ids: List[int]
    disliked_recipe_ids: Optional[List[int]] = []
    days: int = 7

# =========================================================
#                     HELPER FUNCTION
# =========================================================
def compute_user_vector(liked_ids: List[int], disliked_ids: List[int], seq_len: int = 20):
    """
    Compute the Two-Tower user embedding.

    The model takes two sequences:
    - embeddings of liked recipes
    - embeddings of disliked recipes

    Each sequence is padded/truncated to a fixed length.
    """
    embedding_dim = embeddings_df.shape[1]
    
    liked_seq = np.zeros((1, seq_len, embedding_dim), dtype=np.float32)
    disliked_seq = np.zeros((1, seq_len, embedding_dim), dtype=np.float32)
    
    # Keep only recipe IDs present in the embeddings table
    valid_liked = [id for id in liked_ids if id in embeddings_df.index]
    if valid_liked:
        liked_vecs = embeddings_df.loc[valid_liked].values.astype(np.float32)
        n_items = min(len(valid_liked), seq_len)
        liked_seq[0, :n_items] = liked_vecs[-n_items:]
    
    valid_disliked = [id for id in disliked_ids if id in embeddings_df.index]
    if valid_disliked:
        disliked_vecs = embeddings_df.loc[valid_disliked].values.astype(np.float32)
        n_items = min(len(valid_disliked), seq_len)
        disliked_seq[0, :n_items] = disliked_vecs[-n_items:]
    
    # Predict single user embedding vector
    user_vec = user_tower.predict({
        'liked_sequence_input': liked_seq,
        'disliked_sequence_input': disliked_seq
    }, verbose=0)[0]
    
    return user_vec

# =========================================================
#                     API ENDPOINTS
# =========================================================

@app.get("/")
def root():
    """Root endpoint describing API readiness."""
    return {
        "name": "Meal Planner API",
        "version": "3.0.0",
        "status": "ready",
        "models": {
            "user_tower": "loaded",
            "recipe_tower": "loaded",
            "transformer": "loaded"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/stats")
def stats():
    """Return basic dataset statistics."""
    return {
        "recipes": len(recipes_df),
        "embeddings": embeddings_df.shape[0],
        "embedding_dim": embeddings_df.shape[1]
    }

@app.post("/profile")
def calculate_profile(profile: UserProfile):
    """
    Compute caloric needs using:
    - Mifflin-St Jeor BMR formula
    - activity multiplier
    - optional sport calories
    - weight goal adjustment
    """
    # BMR calculation
    if profile.gender.lower() in ["homme", "male", "m"]:
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
    else:
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161
    
    # Activity multipliers
    activity_coeffs = {
        "sedentaire": 1.2,
        "leger": 1.375,
        "modere": 1.55,
        "actif": 1.725,
        "intense": 1.9
    }
    
    coeff = activity_coeffs.get(profile.activity.lower(), 1.2)
    tdee = bmr * coeff
    
    # Additional sport calories (approximation)
    sport_calories = profile.sport_hours * 400 / 7
    tdee += sport_calories
    
    # Weight objective
    if profile.objective == "perte":
        target = tdee - 500
    elif profile.objective == "prise":
        target = tdee + 300
    else:
        target = tdee
    
    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": round(target),
        "bmi": round(profile.weight / (profile.height / 100) ** 2, 1)
    }

@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    """
    Generate AI recipe recommendations using the Two-Tower model.
    """
    if len(request.liked_recipe_ids) == 0:
        raise HTTPException(status_code=400, detail="At least 1 liked recipe is required.")
    
    # User vector from Two-Tower
    user_vec = compute_user_vector(request.liked_recipe_ids, request.disliked_recipe_ids)
    
    # Compute recipe embeddings
    all_emb = embeddings_df.values.astype(np.float32)
    recipe_vecs = recipe_tower.predict(all_emb, batch_size=512, verbose=0)
    
    # Compute cosine similarity
    user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    recipe_norms = recipe_vecs / (np.linalg.norm(recipe_vecs, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(recipe_norms, user_norm)
    
    # Ranking recipes
    exclude = set(request.liked_recipe_ids + request.disliked_recipe_ids)
    sorted_idx = similarities.argsort()[::-1]
    
    recommendations = []
    for idx in sorted_idx:
        recipe_id = embeddings_df.index[idx]
        
        # Skip special tokens
        if isinstance(recipe_id, str) and recipe_id.startswith('<'):
            continue
        
        if recipe_id in exclude:
            continue
        if recipe_id not in recipes_df.index:
            continue
        
        recipe = recipes_df.loc[recipe_id]
        
        # Optional user filters
        if request.max_calories and recipe.get('calories', 0) > request.max_calories:
            continue
        if request.max_time and recipe.get('minutes', 0) > request.max_time:
            continue
        
        rec = recipe.to_dict()
        rec['id'] = int(recipe_id) if isinstance(recipe_id, (int, np.integer)) else recipe_id
        rec['score'] = float(similarities[idx])
        recommendations.append(rec)
        
        if len(recommendations) >= request.n_recommendations:
            break
    
    return {
        "success": True,
        "method": "two_tower",
        "count": len(recommendations),
        "recommendations": recommendations
    }

@app.post("/weekly-planning")
def weekly_planning(request: PlanningRequest):
    """
    Generate a weekly meal plan using the Transformer model.
    """
    if len(request.liked_recipe_ids) == 0:
        raise HTTPException(status_code=400, detail="At least 1 like is required.")
    
    from utils.transformer_generator import generate_weekly_planning
    from utils.token_registry import create_token_registry_from_embeddings
    
    # Compute user embedding
    user_vec = compute_user_vector(request.liked_recipe_ids, request.disliked_recipe_ids)
    
    # Token registry required by Transformer
    token_registry = create_token_registry_from_embeddings(embeddings_df)
    
    # Transformer generation
    planning = generate_weekly_planning(
        user_vec=user_vec,
        recipes_df=recipes_df,
        embeddings_df=embeddings_df,
        category_df=category_df,
        actor_model=transformer_model,
        all_embeddings_df=embeddings_df,
        token_registry=token_registry,
        days=request.days,
        use_transformer=True
    )
    
    return {
        "success": True,
        "method": "transformer",
        "planning": planning
    }

@app.get("/recipe/{recipe_id}")
def get_recipe(recipe_id: int):
    """Fetch recipe details by ID."""
    if recipe_id not in recipes_df.index:
        raise HTTPException(status_code=404, detail="Recipe not found.")
    
    recipe = recipes_df.loc[recipe_id]
    return {
        "id": recipe_id,
        **recipe.to_dict()
    }

# =========================================================
#                     RUN SERVER
# =========================================================
if __name__ == "__main__":
    print("🚀 Starting Meal Planner API...")
    print("📊 Docs available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
