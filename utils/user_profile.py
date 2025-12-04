"""
user_profile.py - Handles user profiles, taste embeddings, and nutritional calculations.
Compatible with the REC_SYS_MAIN_PIPELINE notebook + Transformer (132D vector).
"""

import numpy as np
from typing import Dict, List

# ============================================================
#                NOTEBOOK MEAN VALUES (CRITICAL)
# These normalization constants come directly from the notebook
# They MUST match exactly the values used during model training.
# ============================================================
MEAN_CAL = 2139.656839      # Average daily calories
MEAN_PROT_PCT = 24.403156   # Avg % of calories from protein
MEAN_FAT_PCT = 23.885710    # Avg % of calories from fats
MEAN_CARB_PCT = 51.409447   # Avg % of calories from carbs


# ============================================================
#                     ACTIVITY CONSTANTS
# ============================================================

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9
}

# Simplified mapping used in your Streamlit UI
ACTIVITY_SIMPLE = {
    "sedentary": 1.2,
    "active": 1.55,
    "very_active": 1.9
}


# ============================================================
#                  METABOLIC CALCULATIONS
# ============================================================

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """
    Computes Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation.
    """
    if gender.lower() in ["male", "m", "homme"]:
        s = 5
    else:
        s = -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + s
    return bmr


def calculate_tdee(
    weight: float,
    height: float,
    age: int,
    gender: str,
    activity: str,
    sport_hours: float = 0
) -> float:
    """
    Computes Total Daily Energy Expenditure (TDEE).
    Includes an activity multiplier and optional additional sport calories.
    """
    bmr = calculate_bmr(weight, height, age, gender)

    multiplier = ACTIVITY_SIMPLE.get(activity, 1.2)
    tdee = bmr * multiplier

    # Add calories burned via additional sports (~400 kcal/hour)
    if sport_hours > 0:
        daily_sport_bonus = (sport_hours * 400) / 7
        tdee += daily_sport_bonus

    return round(tdee, 2)


def objective_adjust(tdee: float, objective: str) -> float:
    """
    Adjusts TDEE according to the user's weight goal (simple fixed adjustment).
    """
    adjustments = {
        "weight_loss": -500,
        "muscle_gain": 300,
        "maintenance": 0,
        # Name variants
        "weight loss": -500,
        "muscle gain": 300,
        "lose_weight": -500,
        "gain_muscle": 300,
        "weight_gain": 300
    }

    adjustment = adjustments.get(objective.lower(), 0)
    return round(tdee + adjustment)


def calculate_target_calories(tdee: float, goal: str) -> float:
    """
    Computes caloric target based on goal:
    - Weight loss: -15%
    - Muscle gain: +10%
    - Maintenance: unchanged
    """
    goal = goal.lower()

    if goal == "weight_loss":
        return tdee * 0.85
    elif goal in ["weight_gain", "muscle_gain", "weight gain", "muscle gain", "gain_muscle"]:
        return tdee * 1.10
    else:
        return tdee


def calculate_macros(target_calories: float, goal: str, weight_kg: float) -> Dict[str, float]:
    """
    Computes grams and percentages of macronutrients (protein, fat, carbs).
    Returns:
        protein_g, fat_g, carbs_g, protein_pct, fat_pct, carbs_pct
    """
    goal = goal.lower()

    # Protein calculation based on body weight
    if goal == "weight_loss":
        protein_ratio = 2.0
    elif goal in ["weight_gain", "muscle_gain", "weight gain", "muscle gain", "gain_muscle"]:
        protein_ratio = 1.8
    else:
        protein_ratio = 1.5

    protein_g = weight_kg * protein_ratio

    # Safety limit: max 35% of calories from protein
    max_protein_cal = target_calories * 0.35
    max_protein_g = max_protein_cal / 4
    if protein_g > max_protein_g:
        protein_g = max_protein_g

    protein_cal = protein_g * 4

    # Fats as a percentage of remaining calories
    remaining_cal = target_calories - protein_cal

    fat_share = 0.35 if goal == "weight_loss" else 0.30
    fat_cal = remaining_cal * fat_share
    fat_g = fat_cal / 9

    # Carbs take the remaining calories
    carbs_cal = remaining_cal - fat_cal
    carbs_g = carbs_cal / 4

    # Percentages of total calories
    protein_pct = (protein_cal / target_calories) * 100
    fat_pct = (fat_cal / target_calories) * 100
    carbs_pct = (carbs_cal / target_calories) * 100

    return {
        "protein_g": round(protein_g, 1),
        "fat_g": round(fat_g, 1),
        "carbs_g": round(carbs_g, 1),
        "protein_pct": round(protein_pct, 1),
        "fat_pct": round(fat_pct, 1),
        "carbs_pct": round(carbs_pct, 1)
    }


def calculate_bmi(weight: float, height: float) -> float:
    """
    Computes BMI (Body Mass Index).
    """
    height_m = height / 100
    return round(weight / (height_m ** 2), 1)


def get_bmi_category(bmi: float) -> str:
    """
    Returns the BMI category in English.
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"


# ============================================================
#        USER VECTOR (132D) FOR TRANSFORMER MODEL
# ============================================================

def create_user_vector_132d(
    user_vec_128: np.ndarray,
    target_calories: float,
    macros: Dict[str, float]
) -> np.ndarray:
    """
    Creates the 132D Transformer vector.

    Structure:
    - 0–127: taste embedding (128D Two-Tower)
    - 128 : normalized calories
    - 129 : normalized protein %
    - 130 : normalized fat %
    - 131 : normalized carbs %
    """
    user_cal_day_norm = target_calories / MEAN_CAL
    macro_prot_norm = macros["protein_pct"] / MEAN_PROT_PCT
    macro_fat_norm = macros["fat_pct"] / MEAN_FAT_PCT
    macro_carb_norm = macros["carbs_pct"] / MEAN_CARB_PCT

    user_vec_132 = np.concatenate([
        user_vec_128,
        [user_cal_day_norm, macro_prot_norm, macro_fat_norm, macro_carb_norm]
    ])

    return user_vec_132.astype(np.float32)


def denormalize_user_vector_132d(user_vec: np.ndarray) -> Dict[str, float]:
    """
    Converts a normalized 132D vector back to real caloric + macro values.
    """
    if len(user_vec) < 132:
        raise ValueError(f"Vector too short: {len(user_vec)} < 132")

    target_calories = user_vec[128] * MEAN_CAL
    protein_pct = user_vec[129] * MEAN_PROT_PCT
    fat_pct = user_vec[130] * MEAN_FAT_PCT
    carbs_pct = user_vec[131] * MEAN_CARB_PCT

    return {
        "target_calories": float(target_calories),
        "protein_pct": float(protein_pct),
        "fat_pct": float(fat_pct),
        "carbs_pct": float(carbs_pct)
    }


# ============================================================
#             USER EMBEDDING (Two-Tower model)
# ============================================================

def build_user_embedding_twotower(
    liked_ids: List,
    disliked_ids: List,
    embeddings_df,
    user_tower_model,
    seq_len: int = 20
) -> np.ndarray:
    """
    Builds a 128D user embedding using the Two-Tower model.

    Handles the fact that embeddings_df may contain both:
    - recipe identifiers (numeric)
    - token embeddings (strings)
    """
    if user_tower_model is None:
        raise ValueError("user_tower_model is required")

    # Keep only rows whose index corresponds to a numeric recipe ID
    index_series = embeddings_df.index.to_series()
    numeric_mask = index_series.apply(
        lambda x: isinstance(x, (int, np.integer)) or str(x).isdigit()
    )
    filtered_embeddings = embeddings_df.loc[numeric_mask]

    # Only numeric columns (same as notebook)
    embedding_matrix = filtered_embeddings.select_dtypes(include=[np.number]).values.astype(np.float32)
    num_recipes, embedding_dim = embedding_matrix.shape

    # Map recipe_id → index (exact behavior of notebook)
    raw_ids = filtered_embeddings.index.values
    id_to_idx = {}
    for i, rid in enumerate(raw_ids):
        try:
            id_to_idx[int(rid)] = i
        except (ValueError, TypeError):
            continue

    def create_padded_sequence(vector_list, max_len):
        """
        Pads/truncates embedding sequences exactly like the training notebook.
        """
        if not vector_list:
            return np.zeros((max_len, embedding_dim), dtype=np.float32)

        slice_vectors = vector_list[-max_len:]
        n_items = len(slice_vectors)

        padded_seq = np.zeros((max_len, embedding_dim), dtype=np.float32)
        padded_seq[:n_items] = np.array(slice_vectors)

        return padded_seq

    # Build liked sequence
    liked_vectors = []
    for item_id in liked_ids:
        try:
            clean_id = int(str(item_id).strip())
            idx = id_to_idx.get(clean_id)
            if idx is not None:
                liked_vectors.append(embedding_matrix[idx])
        except (ValueError, TypeError):
            continue

    # Build disliked sequence
    disliked_vectors = []
    for item_id in disliked_ids:
        try:
            clean_id = int(str(item_id).strip())
            idx = id_to_idx.get(clean_id)
            if idx is not None:
                disliked_vectors.append(embedding_matrix[idx])
        except (ValueError, TypeError):
            continue

    # Return zero vector if no valid likes
    if not liked_vectors:
        return np.zeros((128,), dtype=np.float32)

    # Create padded sequences
    liked_seq = create_padded_sequence(liked_vectors, seq_len)
    disliked_seq = create_padded_sequence(disliked_vectors, seq_len)

    # Add batch dimension
    liked_seq = liked_seq.reshape(1, seq_len, embedding_dim)
    disliked_seq = disliked_seq.reshape(1, seq_len, embedding_dim)

    # Predict user embedding
    user_vec = user_tower_model.predict(
        {
            "liked_sequence_input": liked_seq,
            "disliked_sequence_input": disliked_seq
        },
        verbose=0
    )[0]

    return user_vec.astype(np.float32)


def create_user_context(user_vec_128: np.ndarray, profile_embedding_384: np.ndarray) -> np.ndarray:
    """
    Legacy function for 512D context (128D taste + 384D profile).
    Only needed if your Transformer uses that older architecture.
    """
    user_context = np.concatenate([user_vec_128, profile_embedding_384])
    user_context = user_context.reshape(1, 1, -1)
    return user_context.astype(np.float32)


# ============================================================
#             COMPLETE TRANSFORMER USER PROFILE
# ============================================================

def build_complete_user_profile(
    liked_ids: List,
    disliked_ids: List,
    embeddings_df,
    user_tower_model,
    weight: float,
    height: float,
    age: int,
    gender: str,
    activity: str,
    objective: str
) -> Dict:
    """
    Builds a complete user profile ready for the Transformer.

    Pipeline:
        1) Two-Tower → user_vec_128
        2) Compute BMR / TDEE / target calories
        3) Compute macros
        4) Compute BMI
        5) Create final 132D vector with normalized metabolism features
    """

    # 1) Taste embedding (128D)
    user_vec_128 = build_user_embedding_twotower(
        liked_ids=liked_ids,
        disliked_ids=disliked_ids,
        embeddings_df=embeddings_df,
        user_tower_model=user_tower_model,
        seq_len=20
    )

    # 2) Metabolism
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(weight, height, age, gender, activity, sport_hours=0)
    target_calories = calculate_target_calories(tdee, objective)

    # Safety minimum
    min_calories = 1200 if gender.lower() in ["female", "f", "femme"] else 1500
    if target_calories < min_calories:
        target_calories = float(min_calories)

    # 3) Macronutrients
    macros = calculate_macros(target_calories, objective, weight)

    # 4) BMI
    bmi = calculate_bmi(weight, height)

    # 5) Transformer 132D vector
    user_vec_132 = create_user_vector_132d(user_vec_128, target_calories, macros)

    profile = {
        "biometrics": {
            "weight_kg": weight,
            "height_cm": height,
            "age": age,
            "gender": gender
        },
        "activity": {
            "level": activity,
            "goal": objective
        },
        "metabolism": {
            "bmr": round(bmr, 1),
            "tdee": round(tdee, 1),
            "target_calories": round(target_calories, 1)
        },
        # Shortcuts for UI
        "bmr": round(bmr, 1),
        "tdee": round(tdee, 1),
        "target_calories": round(target_calories, 1),
        "target_protein_g": macros["protein_g"],
        "bmi": bmi,
        "macros": macros,
        "user_vec_128": user_vec_128,
        "user_vec_132": user_vec_132,
        "normalization_constants": {
            "mean_cal": MEAN_CAL,
            "mean_prot_pct": MEAN_PROT_PCT,
            "mean_fat_pct": MEAN_FAT_PCT,
            "mean_carb_pct": MEAN_CARB_PCT
        }
    }

    return profile


# ============================================================
#                        TEST FUNCTION
# ============================================================

def test_normalization():
    """Tests that normalization and denormalization are consistent."""
    print("🧪 Testing normalization / denormalization...")

    test_vec_128 = np.random.randn(128).astype(np.float32)
    test_calories = MEAN_CAL
    test_macros = {
        "protein_pct": MEAN_PROT_PCT,
        "fat_pct": MEAN_FAT_PCT,
        "carbs_pct": MEAN_CARB_PCT
    }

    vec_132 = create_user_vector_132d(test_vec_128, test_calories, test_macros)

    print(f"✅ 132D vector created: shape {vec_132.shape}")
    print(f"   Norm features (128–131): {vec_132[128:132]}")
    print("   Expected: [1.0, 1.0, 1.0, 1.0]")

    denorm = denormalize_user_vector_132d(vec_132)

    print("\n✅ Denormalization results:")
    print(f"   Calories: {denorm['target_calories']:.2f} (expected {test_calories:.2f})")
    print(f"   Protein %: {denorm['protein_pct']:.2f}%")
    print(f"   Fat %: {denorm['fat_pct']:.2f}%")
    print(f"   Carbs %: {denorm['carbs_pct']:.2f}%")

    errors = [
        abs(denorm["target_calories"] - test_calories),
        abs(denorm["protein_pct"] - test_macros["protein_pct"]),
        abs(denorm["fat_pct"] - test_macros["fat_pct"]),
        abs(denorm["carbs_pct"] - test_macros["carbs_pct"])
    ]

    if all(e < 0.01 for e in errors):
        print("\n✅ Test PASSED — normalization is consistent.")
    else:
        print(f"\n❌ Test FAILED! Errors: {errors}")


if __name__ == "__main__":
    test_normalization()
