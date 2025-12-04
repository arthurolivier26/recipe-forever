"""
Data Loader — Loading datasets and models
"""

import pandas as pd
import numpy as np
import streamlit as st
import os

# Disable TensorFlow warnings (keeps logs clean)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Required for loading custom Keras 3 models
from utils.custom_layers import CUSTOM_OBJECTS

DATA_DIR = "./data"
MODELS_DIR = "./models"


# ============================================================
#                        DATA LOADING
# ============================================================

@st.cache_data
def load_recipes():
    """Load the main recipes DataFrame."""
    path = f"{DATA_DIR}/recipes_clean.csv"

    if not os.path.exists(path):
        st.error(f"❌ Missing file: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)

    if "id" in df.columns:
        df["id"] = df["id"].astype(int)
        df = df.set_index("id")

    return df


@st.cache_data
def load_meals_category():
    """Load meal categories (Breakfast, Main Dish, Dessert, etc.)."""
    path = f"{DATA_DIR}/meals_category.csv"

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, low_memory=False)

    if "id" in df.columns:
        df = df.set_index("id")

    return df


@st.cache_data
def load_embeddings():
    """
    Load recipe embeddings (typically 384 dimensions).
    Falls back to an alternative file if the main one is missing.
    """
    path = f"{DATA_DIR}/all_embeddings.csv"

    if not os.path.exists(path):
        path = f"{DATA_DIR}/recipes_twotower_embeddings.csv"

    if not os.path.exists(path):
        st.error("❌ Missing embeddings file")
        return None

    df = pd.read_csv(path, index_col=0, low_memory=False)

    # Convert numeric indices while keeping token indices like "<START>"
    def fix_index(idx):
        if isinstance(idx, str) and idx.startswith("<"):
            return idx
        try:
            return int(float(idx))
        except Exception:
            return idx

    df.index = [fix_index(i) for i in df.index]
    df.index.name = "id"

    return df


@st.cache_data
def load_recipe_data():
    """Convenience function. Returns (recipes_df, embeddings_df)."""
    return load_recipes(), load_embeddings()


def load_all_embeddings_with_tokens():
    """Load full embedding table including special tokens."""
    path = f"{DATA_DIR}/all_embeddings.csv"

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, index_col=0, low_memory=False)

    def fix_index(idx):
        if isinstance(idx, str) and idx.startswith("<"):
            return idx
        try:
            return int(float(idx))
        except Exception:
            return idx

    df.index = [fix_index(i) for i in df.index]
    df.index.name = "id"

    return df


# ============================================================
#                          MODEL LOADING
# ============================================================

def _safe_load_model(path):
    """
    Load a model while ensuring compatibility with custom layers/losses.
    Supports both keras.saving.load_model (Keras 3) and tf.keras.
    """
    try:
        import keras
        return keras.saving.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
    except Exception:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            print(f"❌ Error loading model {path}: {e}")
            return None


@st.cache_resource
def load_user_tower():
    """Load the User Tower model from any available file format."""
    paths = [
        f"{MODELS_DIR}/user_tower.keras",
        f"{MODELS_DIR}/trained_user_towers.keras",
        f"{MODELS_DIR}/user_tower.h5",
        f"{MODELS_DIR}/user_tower",
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"⏳ Loading User Tower: {path}")
            model = _safe_load_model(path)
            if model:
                print(f"✅ User Tower loaded from {path}")
                return model

    return None


@st.cache_resource
def load_recipe_tower():
    """Load the Recipe Tower model."""
    paths = [
        f"{MODELS_DIR}/recipe_tower.keras",
        f"{MODELS_DIR}/trained_recipe_towers.keras",
        f"{MODELS_DIR}/recipe_tower.h5",
        f"{MODELS_DIR}/recipe_tower",
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"⏳ Loading Recipe Tower: {path}")
            model = _safe_load_model(path)
            if model:
                print(f"✅ Recipe Tower loaded from {path}")
                return model

    return None


@st.cache_resource
def load_transformer_actor():
    """Load the Transformer Actor model used for planning generation."""
    paths = [
        f"{MODELS_DIR}/actor_final.keras",
        f"{MODELS_DIR}/actor.keras",
        f"{MODELS_DIR}/TRANSFORMER_MODEL_PRETRAIN.keras",
        f"{MODELS_DIR}/actor.h5",
        f"{MODELS_DIR}/actor",
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"⏳ Loading Transformer Actor: {path}")
            model = _safe_load_model(path)
            if model:
                print(f"✅ Transformer loaded from {path}")
                return model

    return None


@st.cache_resource
def load_sentence_transformer():
    """Load the SentenceTransformer encoder for text embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"⚠️ Sentence Transformer unavailable: {e}")
        return None


def load_models():
    """
    Load all available models and return them as a dictionary.
    Missing models are returned as None.
    """
    return {
        "user_tower": load_user_tower(),
        "recipe_tower": load_recipe_tower(),
        "transformer": load_transformer_actor(),
        "sentence_transformer": load_sentence_transformer(),
    }


# ============================================================
#                        UTILITY FUNCTIONS
# ============================================================

def get_recipes_by_category(recipes_df, category_df=None):
    """
    Group recipes by meal category (Breakfast, Main, Dessert, etc.).

    Returns:
        A dictionary where keys are category names and values are filtered DataFrames.
    """
    if category_df is None:
        category_df = load_meals_category()

    if category_df is None:
        return None

    # If categories aren't already in recipes_df, join them
    if "meal_category" not in recipes_df.columns:
        df = recipes_df.join(category_df, how="inner")
    else:
        df = recipes_df

    return {
        "breakfast": df[df["meal_category"] == "Breakfast"],
        "snack": df[df["meal_category"] == "Snack"],
        "starter": df[df["meal_category"] == "Starter"],
        "main": df[df["meal_category"] == "Main Dish"],
        "dessert": df[df["meal_category"] == "Dessert"],
    }


def check_data_availability():
    """
    Check which data files and models are available.
    Useful for debugging or showing status in the UI.
    """
    return {
        "recipes": os.path.exists(f"{DATA_DIR}/recipes_clean.csv"),
        "embeddings": (
            os.path.exists(f"{DATA_DIR}/all_embeddings.csv")
            or os.path.exists(f"{DATA_DIR}/recipes_embeddings.csv")
        ),
        "categories": os.path.exists(f"{DATA_DIR}/meals_category.csv"),
        "user_tower": (
            os.path.exists(f"{MODELS_DIR}/user_tower.keras")
            or os.path.exists(f"{MODELS_DIR}/trained_user_towers.keras")
        ),
        "recipe_tower": (
            os.path.exists(f"{MODELS_DIR}/recipe_tower.keras")
            or os.path.exists(f"{MODELS_DIR}/trained_recipe_towers.keras")
        ),
        "transformer": (
            os.path.exists(f"{MODELS_DIR}/actor_final.keras")
            or os.path.exists(f"{MODELS_DIR}/TRANSFORMER_MODEL_PRETRAIN.keras")
        ),
    }
