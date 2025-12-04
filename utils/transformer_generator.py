"""
Transformer Generator - Weekly planning generation using the notebook system
Compatible with REC_SYS_MAIN_PIPELINE.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.neighbors import NearestNeighbors


# ========== VOCAB MANAGER ==========

class VocabManager:
    """
    Vocabulary manager for the Transformer.
    Handles structural tokens and recipe embeddings.
    """
    
    def __init__(self, wide_embeddings_df, recipes_df):
        """
        Args:
            wide_embeddings_df: DataFrame with embeddings
                                (index = recipe_id, columns = embedding dimensions)
            recipes_df: DataFrame containing recipes with nutrition information
        """
        self.id_to_index = {}
        self.index_to_id = {}
        self.meta_lookup = {}
        
        # Structural tokens required by the Transformer
        self.REQUIRED_TOKENS = [
            "<PAD>", "<START>", "<EOS>",
            "<DAY_START>", "<DAY_END>",
            "<BREAKFAST>", "<LUNCH>", "<SNACK>", "<DINNER>",
            "<STARTER>", "<MAIN_DISH>", "<DESSERT>",
            "<RECIPE_SLOT>"
        ]
        
        # Encode structural tokens into indices
        current_idx = 0
        for token in self.REQUIRED_TOKENS:
            self.id_to_index[token] = current_idx
            self.index_to_id[current_idx] = token
            current_idx += 1
        self.vocab_size_structure = current_idx
        
        # Prepare nutritional data
        print("Preparing nutritional metadata...")
        cols = recipes_df.columns
        prot_col = 'protein pvd' if 'protein pvd' in cols else ('protein' if 'protein' in cols else None)
        cal_col = 'calories' if 'calories' in cols else ('energy' if 'energy' in cols else None)
        
        if prot_col and cal_col:
            nutrition_dict = {str(k): v for k, v in recipes_df[[prot_col, cal_col]].to_dict('index').items()}
        else:
            print("Warning: nutrition columns not found in recipes dataframe.")
            nutrition_dict = {}
        
        # Import recipe vectors
        print(f"Importing {len(wide_embeddings_df)} embedding vectors...")
        count = 0
        
        for recipe_id, row_values in zip(wide_embeddings_df.index, wide_embeddings_df.values):
            # Ignore structural tokens that might be present
            if isinstance(recipe_id, str) and recipe_id.startswith('<'):
                continue
            
            try:
                vec = row_values.astype(np.float32)
            except ValueError:
                continue
            
            # Enforce embedding dimension (384)
            if len(vec) > 384:
                vec = vec[:384]
            if len(vec) < 384:
                continue
            
            # Retrieve nutrition information
            str_id = str(recipe_id)
            nutri = nutrition_dict.get(str_id)
            
            if nutri:
                p = float(nutri.get(prot_col, 0.0))
                c = float(nutri.get(cal_col, 0.0))
            else:
                p, c = 0.0, 0.0
            
            # Store metadata for this recipe ID
            self.meta_lookup[str_id] = {
                "vector": vec,
                "protein": p,
                "energy": c
            }
            count += 1
        
        print(f"{count} recipes successfully imported.")
        
        if count == 0:
            raise ValueError("ERROR: No recipes were imported.")
    
    def get_recipe_data(self, real_id):
        """Return stored metadata for a given recipe ID (vector + nutrition)."""
        return self.meta_lookup.get(str(real_id))
    
    def is_structure_token(self, token):
        """Return True if the token is a structural token (e.g. <BREAKFAST>, <STARTER>, etc.)."""
        return token in self.id_to_index
    
    def encode_structure_token(self, token):
        """Encode a structural token into its integer index (0 if unknown)."""
        return self.id_to_index.get(token, 0)


# ========== HELPER TO CALL RECIPE TOWER WITHOUT WARNINGS ==========

def call_recipe_tower(recipe_tower, rec_emb: np.ndarray) -> np.ndarray:
    """
    Unified call wrapper for the Recipe Tower model, respecting its expected input structure.

    This avoids Keras warnings such as:
    "Expected: ['recipe_vector_input'], Received: Tensor(shape=(1, 384))"
    """
    # Ensure shape (1, 384)
    if rec_emb.ndim == 1:
        rec_emb = np.expand_dims(rec_emb, 0)

    try:
        # If the model has a single named input, Keras usually expects a list [x]
        if hasattr(recipe_tower, "inputs") and isinstance(recipe_tower.inputs, (list, tuple)) and len(recipe_tower.inputs) == 1:
            out = recipe_tower([rec_emb], training=False)
        else:
            out = recipe_tower(rec_emb, training=False)
    except TypeError:
        # Fallback if the "training" argument is not supported
        if hasattr(recipe_tower, "inputs") and isinstance(recipe_tower.inputs, (list, tuple)) and len(recipe_tower.inputs) == 1:
            out = recipe_tower([rec_emb])
        else:
            out = recipe_tower(rec_emb)

    # TensorFlow Eager tensors -> numpy array
    return out.numpy()[0]


# ========== ADAPTIVE DAY SKELETON GENERATION ==========

def generate_adaptive_skeleton(target_kcal: float) -> List[str]:
    """
    Define the high-level meal structure for a day based on calorie budget.

    Args:
        target_kcal: Daily target calorie budget

    Returns:
        List of structural tokens describing the day (e.g. <BREAKFAST>, <RECIPE_SLOT>, ...)
    """
    skeleton = []
    
    # Breakfast (always present)
    skeleton.extend(["<DAY_START>", "<BREAKFAST>", "<RECIPE_SLOT>"])
    
    # Lunch
    skeleton.append("<LUNCH>")
    if target_kcal > 2200:
        skeleton.extend(["<STARTER>", "<RECIPE_SLOT>"])
    skeleton.extend(["<MAIN_DISH>", "<RECIPE_SLOT>"])
    if target_kcal > 1800:
        skeleton.extend(["<DESSERT>", "<RECIPE_SLOT>"])
    
    # Snack (only if budget is high enough)
    if target_kcal > 2000:
        skeleton.extend(["<SNACK>", "<RECIPE_SLOT>"])
    
    # Dinner
    skeleton.append("<DINNER>")
    if target_kcal > 2500:
        skeleton.extend(["<STARTER>", "<RECIPE_SLOT>"])
    skeleton.extend(["<MAIN_DISH>", "<RECIPE_SLOT>"])
    if target_kcal > 1600:
        skeleton.extend(["<DESSERT>", "<RECIPE_SLOT>"])
    
    skeleton.append("<DAY_END>")
    
    return skeleton


# ========== SMART RECIPE SELECTION WITH RERANKING ==========

def select_best_candidate_ultimate(
    target_vector: np.ndarray,
    target_nutri: np.ndarray,
    current_budget_kcal: float,
    target_category: str,
    target_prot_density: float,
    vocab: VocabManager,
    user_pref_norm: np.ndarray,
    recipe_tower,
    knn_engine,
    all_vectors: np.ndarray,
    all_ids_str: List[str],
    recipes_df: pd.DataFrame,
    past_recipes: List[str] = None,
    top_k: int = 150
) -> str:
    """
    Select the best recipe candidate using multi-criteria reranking:
    - similarity in latent space (Transformer prediction vs recipe embedding)
    - calorie budget control
    - protein density target
    - user's taste preference (Two-Tower)
    """
    if past_recipes is None:
        past_recipes = []
    
    # 1. KNN search in embedding space
    dists, indices = knn_engine.kneighbors([target_vector], n_neighbors=top_k)
    
    # Calorie target predicted by the nutrition head
    aim_kcal = target_nutri[1] * 1000
    
    best_id = None
    best_score = -float('inf')
    
    for idx in indices[0]:
        rid = all_ids_str[idx]
        if rid in past_recipes:
            continue
        
        # Strict category filter (meal_category)
        try:
            if rid in recipes_df.index:
                val = recipes_df.loc[rid, 'meal_category']
            elif int(rid) in recipes_df.index:
                val = recipes_df.loc[int(rid), 'meal_category']
            else:
                val = None
            
            # If a target category is specified, enforce matching
            if target_category and val:
                if str(val).strip().lower() != target_category.strip().lower():
                    continue
        except Exception:
            continue
        
        # Retrieve recipe metadata
        data = vocab.get_recipe_data(rid)
        if not data:
            continue
        
        r_kcal = data['energy']
        r_pvd = data['protein']
        if r_kcal < 10:
            # Ignore recipes with extremely low calories
            continue
        
        # Compute protein density (protein calories as % of total calories)
        r_grams = (r_pvd / 100.0) * 50.0
        r_prot_kcal = r_grams * 4.0
        r_density = (r_prot_kcal / r_kcal) * 100 if r_kcal > 0 else 0
        
        # Penalties and constraints
        # 1. Calorie budget penalty
        if current_budget_kcal < 500 and r_kcal > (current_budget_kcal + 100):
            pen_budget = 50.0
        else:
            pen_budget = abs(r_kcal - aim_kcal) / (aim_kcal + 50)
        
        # 2. Protein density penalty
        diff_density = abs(r_density - target_prot_density)
        pen_prot = diff_density * 0.5
        
        # 3. Taste score (via Recipe Tower and user preference)
        rec_emb = np.expand_dims(data['vector'], 0)
        rec_lat = call_recipe_tower(recipe_tower, rec_emb)
        rec_lat = rec_lat / (np.linalg.norm(rec_lat) + 1e-9)
        
        taste = np.dot(user_pref_norm, rec_lat)
        
        # Final scoring function
        final_score = (5.0 * taste) - (2.0 * pen_budget) - (1.0 * pen_prot)
        
        if final_score > best_score:
            best_score = final_score
            best_id = rid
    
    # Fallback: if nothing selected, use the nearest candidate
    if best_id is None:
        best_id = all_ids_str[indices[0][0]]
    
    return best_id


# ========== STRUCTURED PLANNING GENERATION ==========

def generate_planning_structured(
    model,
    vocab: VocabManager,
    user_vec: np.ndarray,
    recipes_df: pd.DataFrame,
    recipe_tower,
    knn_engine,
    all_ids_str: List[str],
    all_vectors: np.ndarray,
    days: int = 7,
    target_daily_kcal: Optional[float] = None,
    target_daily_pvd: Optional[float] = None
) -> Dict:
    """
    Generate a structured weekly planning using the Transformer.

    Args:
        model: Transformer model (actor)
        vocab: VocabManager instance
        user_vec: User vector (132D) consistent with the notebook:
            - 0-127: taste embedding
            - 128: normalized daily calories
            - 129: normalized protein percentage
            - 130: normalized fat percentage
            - 131: normalized carb percentage
        recipes_df: Recipes dataframe
        recipe_tower: Recipe Tower model
        knn_engine: KNN engine built on recipe embeddings
        all_ids_str: List of recipe IDs as strings
        all_vectors: Matrix of recipe vectors (for KNN)
        days: Number of days to generate
        target_daily_kcal: Optional explicit daily calorie target
        target_daily_pvd: Optional explicit daily protein % target

    Returns:
        Dictionary describing the planning with detailed per-day stats.
    """
    
    # Human-readable mapping from structural tokens to meal contexts
    PLAT_TOKENS = {
        "<BREAKFAST>": "Breakfast",
        "<SNACK>": "Snack",
        "<STARTER>": "Starter",
        "<MAIN_DISH>": "Main Dish",
        "<DESSERT>": "Dessert"
    }
    OPTIONAL_TOKENS = ["<STARTER>", "<DESSERT>", "<SNACK>"]
    
    # Compute targets from user_vec (132D)
    CONST_MEAN_CAL = 2145.0  # Used in the original notebook
    
    # Extract normalized factors from the 132D vector (notebook logic)
    if len(user_vec) >= 132:
        # New format: 132D with normalized values
        f_cal = user_vec[128]
        f_prot = user_vec[129]
        f_fat = user_vec[130]
        f_carb = user_vec[131]
    elif len(user_vec) >= 130:
        # Legacy format: 130D (backwards compatibility)
        f_cal = user_vec[128]
        f_prot = 1.0
        f_fat = 1.0
        f_carb = 1.0
    else:
        # Default values if vector is shorter
        f_cal = 1.0
        f_prot = 1.0
        f_fat = 1.0
        f_carb = 1.0
    
    # Denormalize according to the notebook logic
    if target_daily_kcal is None:
        target_daily_kcal = float(f_cal * CONST_MEAN_CAL)
    
    if target_daily_pvd is None:
        # Note: multiplied by 100, not by MEAN_PROT_PCT (follows notebook code)
        target_daily_pvd = float(f_prot * 100.0)
    
    # Compute protein density target (protein kcal as % of daily calories)
    target_prot_grams = (target_daily_pvd / 100.0) * 50.0
    target_prot_kcal = target_prot_grams * 4.0
    target_density_pct = (target_prot_kcal / target_daily_kcal) * 100 if target_daily_kcal > 0 else 0
    
    # Regulation thresholds (for cutting / adding meals)
    THRESHOLD_CUT = target_daily_kcal * 1.05
    THRESHOLD_ADD = target_daily_kcal * 0.85
    
    print("Planning generation")
    print(f"   Target energy: {target_daily_kcal:.0f} kcal")
    print(f"   Target protein: {target_daily_pvd:.0f}% PDV (density: {target_density_pct:.1f}%)")
    print("=" * 60)
    
    # Data structure for full planning
    planning_data = {
        "meta": {
            "target_kcal": target_daily_kcal,
            "target_pvd": target_daily_pvd,
            "target_prot_density": target_density_pct,
            "days_requested": days
        },
        "days": [],
        "global_audit": {}
    }
    
    # Algorithm setup
    daily_skeleton_strs = generate_adaptive_skeleton(3000)  # Always use a rich skeleton, then regulate
    
    user_pref_raw = user_vec[:128]
    user_pref_norm = user_pref_raw / (np.linalg.norm(user_pref_raw) + 1e-9)
    
    # Sequences fed into the Transformer (struct tokens, recipe vectors, budget ratios)
    current_struct_seq = [vocab.encode_structure_token("<START>")]
    current_vect_seq = [np.zeros(384, dtype="float32")]
    current_budget_seq = [[1.0, 1.0]]
    
    past_recipes = []
    stats_history = {'kcal': [], 'pvd': [], 'taste': [], 'density': []}
    
    RECIPE_SLOT_ID = vocab.encode_structure_token("<RECIPE_SLOT>")
    RECIPE_SLOT_STR = "<RECIPE_SLOT>"
    current_dish_context = None
    skip_next_recipe_slot = False
    
    # Day loop
    for day_idx in range(1, days + 1):
        day_object = {
            "day_index": day_idx,
            "steps": [],
            "stats": {}
        }
        
        day_kcal = 0.0
        day_pvd = 0.0
        day_prot_kcal = 0.0
        
        for step_token_str in daily_skeleton_strs:
            
            # Regulation for optional dishes (starter, dessert, snack)
            if step_token_str in OPTIONAL_TOKENS:
                if day_kcal >= THRESHOLD_CUT:
                    skip_next_recipe_slot = True
                    day_object["steps"].append({
                        "type": "regulation_cut",
                        "token": step_token_str,
                        "reason": "budget_exceeded",
                        "current_kcal": float(day_kcal)
                    })
                    continue
                else:
                    skip_next_recipe_slot = False
            
            if step_token_str == RECIPE_SLOT_STR and skip_next_recipe_slot:
                skip_next_recipe_slot = False
                continue
            
            # Update dish context (Breakfast, Starter, Main Dish, Dessert, Snack)
            if step_token_str in PLAT_TOKENS:
                current_dish_context = PLAT_TOKENS[step_token_str]
            
            # Build historical inputs for the Transformer
            hist_struct = current_struct_seq[-141:]
            hist_vect = current_vect_seq[-141:]
            hist_budget = current_budget_seq[-141:]
            
            inp_struct = np.zeros((1, 141), dtype="int32")
            inp_struct[0, :len(hist_struct)] = hist_struct
            
            inp_vect = np.zeros((1, 141, 384), dtype="float32")
            inp_vect[0, :len(hist_vect)] = hist_vect
            
            inp_budget = np.zeros((1, 141, 2), dtype="float32")
            inp_budget[0, :len(hist_budget)] = hist_budget
            
            inp_user = np.expand_dims(user_vec, axis=0)
            
            # Recipe slot case: ask the Transformer + reranker which recipe to insert
            if step_token_str == RECIPE_SLOT_STR:
                preds = model.predict([inp_user, inp_struct, inp_vect, inp_budget], verbose=0)
                last_idx = len(hist_struct) - 1
                pred_vec = preds[1][0, last_idx, :]
                pred_nutri = preds[2][0, last_idx, :]
                
                rem_kcal = max(0.0, target_daily_kcal - day_kcal)
                
                # Multi-criteria reranking to choose the final recipe
                best_id = select_best_candidate_ultimate(
                    target_vector=pred_vec,
                    target_nutri=pred_nutri,
                    current_budget_kcal=rem_kcal,
                    target_category=current_dish_context,
                    target_prot_density=target_density_pct,
                    vocab=vocab,
                    user_pref_norm=user_pref_norm,
                    recipe_tower=recipe_tower,
                    knn_engine=knn_engine,
                    all_vectors=all_vectors,
                    all_ids_str=all_ids_str,
                    recipes_df=recipes_df,
                    past_recipes=past_recipes,
                    top_k=150
                )
                
                past_recipes.append(best_id)
                current_struct_seq.append(RECIPE_SLOT_ID)
                real_data = vocab.get_recipe_data(best_id)
                current_vect_seq.append(real_data['vector'])
                
                # Nutrition stats for this recipe
                r_kcal = float(real_data['energy'])
                r_pvd = float(real_data['protein'])
                r_grams = (r_pvd / 100.0) * 50.0
                r_pkcal = r_grams * 4.0
                r_dens = (r_pkcal / r_kcal * 100.0) if r_kcal > 0 else 0.0
                
                day_kcal += r_kcal
                day_pvd += r_pvd
                day_prot_kcal += r_pkcal
                
                new_ratio = max(0.0, (target_daily_kcal - day_kcal) / target_daily_kcal)
                current_budget_seq.append([new_ratio, new_ratio])
                
                # Taste score using Recipe Tower projection
                rec_emb = np.expand_dims(real_data['vector'], 0)
                rec_lat = call_recipe_tower(recipe_tower, rec_emb)
                rec_lat = rec_lat / (np.linalg.norm(rec_lat) + 1e-9)
                taste = float(np.dot(user_pref_norm, rec_lat))
                stats_history['taste'].append(taste)
                
                # Retrieve recipe name
                try:
                    if int(best_id) in recipes_df.index:
                        r_name = recipes_df.loc[int(best_id), 'name']
                    else:
                        r_name = recipes_df.loc[best_id, 'name']
                except Exception:
                    r_name = "Unknown recipe"
                
                # Append recipe step
                day_object["steps"].append({
                    "type": "recipe",
                    "id": str(best_id),
                    "name": str(r_name),
                    "category_context": current_dish_context,
                    "nutrition": {
                        "kcal": r_kcal,
                        "pvd": r_pvd,
                        "prot_density": r_dens
                    },
                    "metrics": {
                        "taste_score": taste,
                        "is_added_dynamically": False
                    }
                })
            
            # Structural token case (e.g. <BREAKFAST>, <LUNCH>, <STARTER>, etc.)
            else:
                token_id = vocab.encode_structure_token(step_token_str)
                current_struct_seq.append(token_id)
                current_vect_seq.append(np.zeros(384, dtype="float32"))
                current_budget_seq.append(current_budget_seq[-1])
                
                # Only log "structure_token" events that are not DAY/RECIPE markers
                if step_token_str.startswith("<") and "RECIPE" not in step_token_str and "DAY" not in step_token_str:
                    day_object["steps"].append({
                        "type": "structure_token",
                        "token": step_token_str
                    })
        
        # Calorie catch-up if the day is far below the target (add a snack)
        if day_kcal < THRESHOLD_ADD:
            missing = target_daily_kcal - day_kcal
            
            hist_struct = current_struct_seq[-141:]
            hist_vect = current_vect_seq[-141:]
            hist_budget = current_budget_seq[-141:]
            
            inp_struct = np.zeros((1, 141), dtype="int32")
            inp_struct[0, :len(hist_struct)] = hist_struct
            
            inp_vect = np.zeros((1, 141, 384), dtype="float32")
            inp_vect[0, :len(hist_vect)] = hist_vect
            
            inp_budget = np.zeros((1, 141, 2), dtype="float32")
            inp_budget[0, :len(hist_budget)] = hist_budget
            inp_budget[0, -1, :] = [missing / target_daily_kcal, missing / target_daily_kcal]
            
            preds = model.predict([inp_user, inp_struct, inp_vect, inp_budget], verbose=0)
            p_vec = preds[1][0, -1, :]
            p_nut = preds[2][0, -1, :]
            
            snack_id = select_best_candidate_ultimate(
                target_vector=p_vec,
                target_nutri=p_nut,
                current_budget_kcal=missing,
                target_category="Snack",
                target_prot_density=20.0,
                vocab=vocab,
                user_pref_norm=user_pref_norm,
                recipe_tower=recipe_tower,
                knn_engine=knn_engine,
                all_vectors=all_vectors,
                all_ids_str=all_ids_str,
                recipes_df=recipes_df,
                past_recipes=past_recipes,
                top_k=150
            )
            
            d_snack = vocab.get_recipe_data(snack_id)
            past_recipes.append(snack_id)
            
            s_kcal = float(d_snack['energy'])
            s_pvd = float(d_snack['protein'])
            s_dens = ((s_pvd / 100.0 * 50.0 * 4.0) / s_kcal * 100.0) if s_kcal > 0 else 0.0
            
            day_kcal += s_kcal
            day_pvd += s_pvd
            
            try:
                if int(snack_id) in recipes_df.index:
                    s_name = recipes_df.loc[int(snack_id), 'name']
                else:
                    s_name = recipes_df.loc[snack_id, 'name']
            except Exception:
                s_name = "Snack"
            
            day_object["steps"].append({
                "type": "recipe",
                "id": str(snack_id),
                "name": str(s_name),
                "category_context": "Snack",
                "nutrition": {
                    "kcal": s_kcal,
                    "pvd": s_pvd,
                    "prot_density": s_dens
                },
                "metrics": {
                    "taste_score": 0.0,
                    "is_added_dynamically": True,
                    "reason": "caloric_deficit_fill"
                }
            })
        
        # Daily summary statistics
        day_dens = (day_prot_kcal / day_kcal * 100.0) if day_kcal > 0 else 0.0
        
        day_object["stats"] = {
            "total_kcal": float(day_kcal),
            "total_pvd": float(day_pvd),
            "mean_density": float(day_dens),
            "delta_target_kcal": float(day_kcal - target_daily_kcal)
        }
        
        planning_data["days"].append(day_object)
        
        stats_history['kcal'].append(day_kcal)
        stats_history['pvd'].append(day_pvd)
        stats_history['density'].append(day_dens)
    
    # Global audit across days (energy, protein, taste)
    if len(stats_history['kcal']) > 0:
        avg_k = float(np.mean(stats_history['kcal']))
        std_k = float(np.std(stats_history['kcal']))
        avg_p = float(np.mean(stats_history['pvd']))
        avg_d = float(np.mean(stats_history['density']))
        avg_t = float(np.mean(stats_history['taste'])) if stats_history['taste'] else 0.0
        
        planning_data["global_audit"] = {
            "energy": {
                "mean_kcal": avg_k,
                "std_dev_kcal": std_k,
                "delta_target": avg_k - target_daily_kcal,
                "percent_error": ((avg_k - target_daily_kcal) / target_daily_kcal) * 100.0
            },
            "protein": {
                "mean_pvd": avg_p,
                "mean_density": avg_d,
                "delta_density": avg_d - target_density_pct
            },
            "taste": {
                "mean_score": avg_t,
                "verdict": "excellent" if avg_t > 0.8 else "good" if avg_t > 0.6 else "average"
            }
        }
    
    return planning_data


# ========== WRAPPER FUNCTION (LEGACY API COMPATIBLE) ==========

def generate_weekly_planning(
    user_vec: np.ndarray,
    recipes_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    category_df: pd.DataFrame,
    actor_model,
    all_embeddings_df: pd.DataFrame,
    token_registry=None,
    days: int = 7,
    use_transformer: bool = True,
    target_calories: Optional[float] = None
) -> Dict:
    """
    Wrapper function compatible with the legacy API.
    Handles vocabulary & KNN setup, then calls the Transformer planner.
    """
    if not use_transformer:
        raise NotImplementedError("Only Transformer-based generation is supported.")
    
    # Merge recipes dataframe with categories if needed
    if 'meal_category' not in recipes_df.columns and category_df is not None:
        # Keep only the 'meal_category' column to avoid column conflicts
        if 'meal_category' in category_df.columns:
            category_col = category_df[['meal_category']].copy()
            recipes_df = recipes_df.join(category_col, how='left')
        else:
            recipes_df = recipes_df.join(category_df, how='left', rsuffix='_cat')
    
    # Initialize vocabulary manager
    print("Initializing VocabManager...")
    vocab = VocabManager(all_embeddings_df, recipes_df)
    
    # Build KNN index on recipe vectors
    print("Building KNN index...")
    all_ids_str = list(vocab.meta_lookup.keys())
    all_vectors = np.array([vocab.meta_lookup[id]['vector'] for id in all_ids_str])
    
    knn_engine = NearestNeighbors(
        n_neighbors=min(150, len(all_vectors)),
        metric='cosine',
        algorithm='brute'
    )
    knn_engine.fit(all_vectors)
    
    # Load Recipe Tower using data_loader helper
    from utils.data_loader import load_recipe_tower
    recipe_tower = load_recipe_tower()
    
    if recipe_tower is None:
        raise ValueError("Recipe Tower is not available.")
    
    # Generate raw planning structure
    planning_dict = generate_planning_structured(
        model=actor_model,
        vocab=vocab,
        user_vec=user_vec,
        recipes_df=recipes_df,
        recipe_tower=recipe_tower,
        knn_engine=knn_engine,
        all_ids_str=all_ids_str,
        all_vectors=all_vectors,
        days=days,
        target_daily_kcal=target_calories
    )
    
    # Convert into the UI-friendly format
    return convert_to_ui_format(planning_dict, recipes_df)


def convert_to_ui_format(planning_dict: Dict, recipes_df: pd.DataFrame) -> Dict:
    """
    Convert the structured planning into the format expected by the UI.

    Output structure:
    {
        "days": [
            {
                "day": <int>,
                "meals": [
                    {
                        "type": "breakfast" | "lunch" | "snack" | "dinner",
                        "courses": [
                            {
                                "type": "starter" | "main" | "dessert",
                                "recipe": { ... }
                            }
                        ]
                    },
                    ...
                ],
                "stats": { ... }
            },
            ...
        ],
        "stats": {
            "num_days": ...,
            "total_recipes": ...,
            "avg_calories_per_day": ...
        }
    }
    """
    ui_format = {
        "days": [],
        "stats": {
            "num_days": planning_dict["meta"]["days_requested"],
            "total_recipes": 0,
            "avg_calories_per_day": 0
        }
    }
    
    total_recipes = 0
    total_calories = 0.0
    
    for day_data in planning_dict["days"]:
        day_num = day_data["day_index"]
        
        # Organize recipes by meal type
        meals_dict = {
            'breakfast': {'type': 'breakfast', 'courses': []},
            'lunch': {'type': 'lunch', 'courses': []},
            'snack': {'type': 'snack', 'courses': []},
            'dinner': {'type': 'dinner', 'courses': []}
        }
        
        current_meal = None
        
        for step in day_data["steps"]:
            if step["type"] == "structure_token":
                token = step["token"]
                if token == "<BREAKFAST>":
                    current_meal = 'breakfast'
                elif token == "<LUNCH>":
                    current_meal = 'lunch'
                elif token == "<SNACK>":
                    current_meal = 'snack'
                elif token == "<DINNER>":
                    current_meal = 'dinner'
            
            elif step["type"] == "recipe" and current_meal:
                recipe_id = int(step["id"])
                
                # Retrieve full row if possible (for minutes, etc.)
                try:
                    if recipe_id in recipes_df.index:
                        recipe_row = recipes_df.loc[recipe_id]
                    else:
                        recipe_row = None
                except Exception:
                    recipe_row = None
                
                # Normalize course type to starter/main/dessert
                course_type = step.get("category_context", "main").lower()
                if "starter" in course_type:
                    course_type = "starter"
                elif "dessert" in course_type:
                    course_type = "dessert"
                else:
                    course_type = "main"
                
                course = {
                    "type": course_type,
                    "recipe": {
                        "id": recipe_id,
                        "name": step["name"],
                        "calories": int(step["nutrition"]["kcal"]),
                        "minutes": int(recipe_row.get('minutes', 0)) if recipe_row is not None else 0,
                        "protein": step["nutrition"]["pvd"],
                        "taste_score": step["metrics"]["taste_score"]
                    }
                }
                
                meals_dict[current_meal]['courses'].append(course)
                total_recipes += 1
        
        # Keep only non-empty meals
        meals = [m for m in meals_dict.values() if len(m['courses']) > 0]
        
        ui_format["days"].append({
            "day": day_num,
            "meals": meals,
            "stats": day_data["stats"]
        })
        
        total_calories += day_data["stats"]["total_kcal"]
    
    ui_format["stats"]["total_recipes"] = total_recipes
    if len(planning_dict["days"]) > 0:
        ui_format["stats"]["avg_calories_per_day"] = int(total_calories / len(planning_dict["days"]))
    else:
        ui_format["stats"]["avg_calories_per_day"] = 0
    
    return ui_format
