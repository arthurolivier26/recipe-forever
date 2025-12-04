"""
Recommender — Two-Tower Recommendation System
Compatible with the REC_SYS_MAIN_PIPELINE notebook
FINAL VERSION — Includes Temperature Scaling + Sigmoid transformation
"""

import numpy as np
from typing import List, Optional, Tuple
import pandas as pd


# ============================================================
#               TEMPERATURE SCALING + SIGMOID
# ============================================================

def apply_temperature_sigmoid(similarities: np.ndarray, temperature: float = 5.0) -> np.ndarray:
    """
    Apply Temperature Scaling followed by a Sigmoid transformation.
    This matches exactly the post-processing used in the Two-Tower notebook.

    Args:
        similarities: Raw cosine similarities (typically between 0 and 1).
        temperature: Multiplicative scaling factor before applying the sigmoid.
                     Default is 5.0, as used in the notebook.

    Returns:
        A transformed score array in the range [0, 1].
        Good recommendations usually fall in the 0.85–0.99 range.
    """
    # 1. Temperature Scaling
    scaled = similarities * temperature

    # 2. Sigmoid transform
    scores = 1 / (1 + np.exp(-scaled))

    return scores


# ============================================================
#                     SIMILARITY FUNCTIONS
# ============================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def batch_cosine_similarity(user_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector and a matrix of vectors.
    Designed to scale efficiently with large datasets.

    Args:
        user_vec: User embedding (D,)
        matrix: Matrix of recipe embeddings (N, D)

    Returns:
        An array of similarities of shape (N,)
    """
    # L2-normalization
    user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    # Dot product equals cosine similarity after normalization
    similarities = np.dot(matrix_norms, user_norm)

    return similarities


# ============================================================
#                        TOP-K RECIPES
# ============================================================

def top_k_recipes(
    user_vec: np.ndarray,
    recipes_df,
    embeddings_df,
    recipe_tower_model=None,
    k: int = 10,
    exclude_ids: Optional[List[int]] = None,
    use_temperature: bool = True
):
    """
    Return the Top-K most similar recipes to the given user vector.

    Args:
        user_vec: User embedding (either 128D or 384D)
        recipes_df: DataFrame containing recipe metadata
        embeddings_df: DataFrame containing recipe embeddings
        recipe_tower_model: Optional projection model (384D -> 128D)
        k: Number of recommendations to return
        exclude_ids: Optional list of recipe IDs to ignore
        use_temperature: Whether to apply Temperature Scaling + Sigmoid

    Returns:
        A DataFrame containing the recommended recipes and their scores.
    """
    if exclude_ids is None:
        exclude_ids = []

    exclude_set = set(exclude_ids)

    # If a Recipe Tower exists, project all recipe embeddings to 128D
    if recipe_tower_model is not None:
        all_embeddings = embeddings_df.select_dtypes(include=[np.number]).values.astype(np.float32)
        recipe_vectors = recipe_tower_model.predict(all_embeddings, batch_size=512, verbose=0)
    else:
        recipe_vectors = embeddings_df.select_dtypes(include=[np.number]).values.astype(np.float32)

    # Compute cosine similarities
    user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    recipe_norms = recipe_vectors / (np.linalg.norm(recipe_vectors, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(recipe_norms, user_norm)

    # Optional: apply Temperature Scaling
    final_scores = apply_temperature_sigmoid(similarities, temperature=5.0) if use_temperature else similarities

    # Retrieve the top-K results
    sorted_indices = final_scores.argsort()[::-1]

    results = []
    for idx in sorted_indices:
        recipe_id = embeddings_df.index[idx]

        # Skip special tokens such as "<START>"
        if isinstance(recipe_id, str) and recipe_id.startswith("<"):
            continue

        if recipe_id in exclude_set:
            continue

        if recipe_id not in recipes_df.index:
            continue

        recipe = recipes_df.loc[recipe_id]
        recipe_dict = recipe.to_dict()
        recipe_dict["recipe_id"] = recipe_id
        recipe_dict["score"] = float(final_scores[idx])
        results.append(recipe_dict)

        if len(results) >= k:
            break

    return pd.DataFrame(results)


# ============================================================
#              FULL TWO-TOWER RECOMMENDATION PIPELINE
# ============================================================

def recommend_with_twotower(
    liked_ids: List[int],
    disliked_ids: List[int],
    embeddings_df,
    recipes_df,
    user_tower_model,
    recipe_tower_model,
    k: int = 10,
    seq_len: int = 20,
    use_temperature: bool = True
):
    """
    Generate recommendations using the complete Two-Tower architecture.

    Args:
        liked_ids: IDs of positively-rated recipes
        disliked_ids: IDs of negatively-rated recipes
        embeddings_df: Embedding matrix of all recipes (384D)
        recipes_df: Metadata about recipes
        user_tower_model: Model encoding user preferences
        recipe_tower_model: Model encoding recipe representations
        k: Number of recommendations
        seq_len: Maximum sequence length for the User Tower
        use_temperature: Whether to apply Temperature Scaling + Sigmoid

    Returns:
        - A DataFrame containing the Top-K recommendations
        - The user vector (128D)
    """
    embedding_dim = embeddings_df.select_dtypes(include=[np.number]).shape[1]  # typically 384

    # 1. Build padded input sequences
    liked_seq = np.zeros((1, seq_len, embedding_dim), dtype=np.float32)
    disliked_seq = np.zeros((1, seq_len, embedding_dim), dtype=np.float32)

    valid_liked = [id for id in liked_ids if id in embeddings_df.index]
    valid_disliked = [id for id in disliked_ids if id in embeddings_df.index]

    # Fill liked sequence
    if valid_liked:
        liked_vecs = embeddings_df.loc[valid_liked].select_dtypes(include=[np.number]).values.astype(np.float32)
        n_items = min(len(valid_liked), seq_len)
        liked_seq[0, :n_items] = liked_vecs[-n_items:]

    # Fill disliked sequence
    if valid_disliked:
        disliked_vecs = embeddings_df.loc[valid_disliked].select_dtypes(include=[np.number]).values.astype(np.float32)
        n_items = min(len(valid_disliked), seq_len)
        disliked_seq[0, :n_items] = disliked_vecs[-n_items:]

    # 2. Generate 128D user embedding
    user_vec = user_tower_model.predict({
        "liked_sequence_input": liked_seq,
        "disliked_sequence_input": disliked_seq
    }, verbose=0)[0]

    # 3. Project all recipes using the Recipe Tower (384D → 128D)
    all_embeddings = embeddings_df.select_dtypes(include=[np.number]).values.astype(np.float32)
    recipe_vectors = recipe_tower_model.predict(all_embeddings, batch_size=512, verbose=0)

    # 4. Compute similarities in the 128D space
    user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    recipe_norms = recipe_vectors / (np.linalg.norm(recipe_vectors, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(recipe_norms, user_norm)

    # Post-processing step: Temperature Scaling
    final_scores = apply_temperature_sigmoid(similarities, temperature=5.0) if use_temperature else similarities

    # 5. Top-K filtering
    exclude_set = set(liked_ids + disliked_ids)
    sorted_indices = final_scores.argsort()[::-1]

    results = []
    for idx in sorted_indices:
        recipe_id = embeddings_df.index[idx]

        if isinstance(recipe_id, str) and recipe_id.startswith("<"):
            continue

        if recipe_id in exclude_set:
            continue

        if recipe_id not in recipes_df.index:
            continue

        recipe = recipes_df.loc[recipe_id]
        recipe_dict = recipe.to_dict()
        recipe_dict["recipe_id"] = recipe_id
        recipe_dict["score"] = float(final_scores[idx])
        results.append(recipe_dict)

        if len(results) >= k:
            break

    return pd.DataFrame(results), user_vec


# ============================================================
#                  SIMILARITY SEARCH BETWEEN RECIPES
# ============================================================

def get_similar_recipes(
    recipe_id: int,
    embeddings_df,
    recipes_df,
    recipe_tower_model=None,
    k: int = 5,
    use_temperature: bool = True
):
    """
    Retrieve recipes similar to a given recipe using cosine similarity.

    Args:
        recipe_id: ID of the reference recipe
        embeddings_df: DataFrame of recipe embeddings (384D)
        recipes_df: Recipe metadata
        recipe_tower_model: Optional projection model (384D → 128D)
        k: Number of similar recipes to return
        use_temperature: Whether to apply Temperature Scaling

    Returns:
        A DataFrame containing similar recipes.
    """
    if recipe_id not in embeddings_df.index:
        return None

    # Extract original 384D embedding
    source_vec_384 = embeddings_df.loc[recipe_id].select_dtypes(include=[np.number]).values

    # Optional projection through Recipe Tower
    if recipe_tower_model is not None:
        source_vec = recipe_tower_model.predict(source_vec_384.reshape(1, -1), verbose=0)[0]
    else:
        source_vec = source_vec_384

    return top_k_recipes(
        user_vec=source_vec,
        recipes_df=recipes_df,
        embeddings_df=embeddings_df,
        recipe_tower_model=recipe_tower_model,
        k=k + 1,  # +1 to compensate for the source recipe being in the results
        exclude_ids=[recipe_id],
        use_temperature=use_temperature
    )


# ============================================================
#                   CATEGORY-BASED RECOMMENDATIONS
# ============================================================

def recommend_by_category(
    user_vec: np.ndarray,
    category: str,
    recipes_df,
    embeddings_df,
    category_df,
    recipe_tower_model=None,
    k: int = 5,
    exclude_ids: Optional[List[int]] = None,
    use_temperature: bool = True
):
    """
    Generate recommendations restricted to a specific meal category.

    Args:
        user_vec: User embedding (128D)
        category: Target meal category ("Breakfast", "Main Dish", "Dessert", etc.)
        recipes_df: All recipe metadata
        embeddings_df: Embedding matrix (384D)
        category_df: DataFrame mapping recipe_id → category
        recipe_tower_model: Optional projection model
        k: Number of results
        exclude_ids: Recipe IDs to ignore
        use_temperature: Whether to apply Temperature Scaling

    Returns:
        A DataFrame of recommendations limited to the requested category.
    """
    if exclude_ids is None:
        exclude_ids = []

    # Filter recipes belonging to the requested category
    if "meal_category" in recipes_df.columns:
        category_recipes = recipes_df[recipes_df["meal_category"] == category]
    else:
        # If the category is stored elsewhere, join the two DataFrames
        recipes_with_cat = recipes_df.join(category_df, how="inner")
        category_recipes = recipes_with_cat[recipes_with_cat["meal_category"] == category]

    if len(category_recipes) == 0:
        return None

    # Select embeddings for valid recipes in this category
    valid_ids = [rid for rid in category_recipes.index if rid in embeddings_df.index]
    category_embeddings = embeddings_df.loc[valid_ids]

    return top_k_recipes(
        user_vec=user_vec,
        recipes_df=category_recipes,
        embeddings_df=category_embeddings,
        recipe_tower_model=recipe_tower_model,
        k=k,
        exclude_ids=exclude_ids,
        use_temperature=use_temperature
    )
