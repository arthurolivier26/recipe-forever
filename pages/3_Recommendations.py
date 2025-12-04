"""
3_Recommendations.py — Personalized recipe recommendation page
"""

import streamlit as st
import numpy as np
import os

# ============================================================
#                         PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Recommendations - Meal Planner AI",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
#                      MODEL VERIFICATION
# ============================================================
if not os.path.exists("./models/trained_user_towers.keras"):
    st.error("⛔ Missing models. Returning to home page.")
    st.page_link("Home.py", label="🏠 Home", icon="🏠")
    st.stop()

from utils.ui_components import inject_css, recipe_card
from utils.navigation import init_session_state, sidebar_status
from utils.data_loader import (
    load_recipe_data, load_embeddings,
    load_user_tower, load_recipe_tower
)
from utils.recommender import recommend_with_twotower
from utils.user_profile import build_user_embedding_twotower

inject_css()
init_session_state()

# ============================================================
#                         LOAD DATA
# ============================================================
recipes_df, embeddings_df = load_recipe_data()

# ============================================================
#                           HEADER
# ============================================================
st.markdown("# 🎯 Your Personalized Recommendations")

if recipes_df is None or embeddings_df is None:
    st.error("❌ Data files not available. Please check your CSV files.")
    st.stop()

# Load models
user_tower = load_user_tower()
recipe_tower = load_recipe_tower()

if user_tower is None or recipe_tower is None:
    st.error("❌ Two-Tower models could not be loaded. Check your .keras files.")
    st.stop()

# Require at least 5 likes for meaningful recommendations
if len(st.session_state.likes) < 5:
    st.warning(
        f"⚠️ You have liked only {len(st.session_state.likes)} recipes. "
        "Like at least 5 recipes for better recommendations."
    )

    if st.button("❤️ Start swiping", type="primary"):
        st.switch_page("pages/2_Swipe.py")
    st.stop()

st.markdown(
    f"Based on your **{len(st.session_state.likes)} likes** "
    f"and **{len(st.session_state.dislikes)} dislikes**"
)

st.markdown("---")

# ============================================================
#                           FILTERS
# ============================================================
st.markdown("### 🔍 Filters")

col1, col2, col3 = st.columns(3)

with col1:
    n_recommendations = st.slider("Number of recommendations", 5, 50, 15)

with col2:
    max_calories = st.slider("Max calories", 100, 2000, 1500)

with col3:
    max_time = st.slider("Max cooking time (min)", 10, 300, 120)

st.markdown("---")

# ============================================================
#                  GENERATE RECOMMENDATIONS
# ============================================================
if st.button("🚀 Generate my recommendations", type="primary", use_container_width=True):

    with st.spinner("🧠 AI is analyzing your taste profile using the Two-Tower model..."):

        try:
            recommendations, user_vec = recommend_with_twotower(
                liked_ids=st.session_state.likes,
                disliked_ids=st.session_state.dislikes,
                embeddings_df=embeddings_df,
                recipes_df=recipes_df,
                user_tower_model=user_tower,
                recipe_tower_model=recipe_tower,
                k=n_recommendations * 2  # generate more then filter
            )

            st.session_state.user_vec = user_vec

            # Apply user filters
            if recommendations is not None and len(recommendations) > 0:
                if "calories" in recommendations.columns:
                    recommendations = recommendations[recommendations["calories"] <= max_calories]
                if "minutes" in recommendations.columns:
                    recommendations = recommendations[recommendations["minutes"] <= max_time]

                recommendations = recommendations.head(n_recommendations)

            st.session_state.recommendations = recommendations

        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")
            st.session_state.recommendations = None

# ============================================================
#                     DISPLAY RECOMMENDATIONS
# ============================================================
if st.session_state.get("recommendations") is not None and len(st.session_state.recommendations) > 0:

    recommendations = st.session_state.recommendations

    st.success(f"✅ {len(recommendations)} recommendations generated using the **Two-Tower** model!")
    st.markdown("---")

    # ---------------------- Stats section ----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_cal = int(recommendations["calories"].mean()) if "calories" in recommendations.columns else 0
        st.metric("🔥 Avg. calories", f"{avg_cal} kcal")

    with col2:
        avg_time = int(recommendations["minutes"].mean()) if "minutes" in recommendations.columns else 0
        st.metric("⏱️ Avg. time", f"{avg_time} min")

    with col3:
        avg_score = recommendations["score"].mean() if "score" in recommendations.columns else 0
        st.metric("🎯 Avg. match score", f"{avg_score:.2f}")

    st.markdown("---")

    # ---------------------- Recipe grid ----------------------
    st.markdown("### 🍽️ Recommended Recipes")

    cols = st.columns(3)

    for i, (idx, recipe) in enumerate(recommendations.iterrows()):
        with cols[i % 3]:
            score = recipe.get("score", 0)
            score_pct = int(score * 100)

            # Score badge style
            if score > 0.8:
                score_color = "#3fb950"
                score_emoji = "🔥"
            elif score > 0.6:
                score_color = "#f0883e"
                score_emoji = "👍"
            else:
                score_color = "#8b949e"
                score_emoji = "🤔"

            st.markdown(f"""
            <div class="recipe-card">
                <div class="recipe-title">{recipe.get('name', 'Recipe')}</div>
                <div class="recipe-meta">
                    🔥 {int(recipe.get('calories', 0))} kcal &nbsp;|&nbsp;
                    ⏱️ {int(recipe.get('minutes', 0))} min
                </div>
                <div style="margin-top: 10px;">
                    <span class="badge badge-green">{score_emoji} Match: {score_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expander with details
            with st.expander("📋 Details"):
                st.write(f"**ID:** {recipe.get('recipe_id', idx)}")

                if "n_ingredients" in recipe:
                    st.write(f"**Ingredients:** {int(recipe['n_ingredients'])}")

                if "n_steps" in recipe:
                    st.write(f"**Steps:** {int(recipe['n_steps'])}")

                col_a, col_b = st.columns(2)
                with col_a:
                    protein = recipe.get("protein pvd", recipe.get("protein", 0))
                    st.caption(f"🥩 Protein: {protein}%")
                with col_b:
                    carbs = recipe.get("carbohydrates pvd", recipe.get("carbs", 0))
                    st.caption(f"🍚 Carbs: {carbs}%")

    st.markdown("---")

    # ---------------------- Action buttons ----------------------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.recommendations = None
            st.rerun()

    with col2:
        if st.button("📅 Generate weekly plan", type="primary", use_container_width=True):
            st.switch_page("pages/4_Weekly_Planning.py")

elif st.session_state.get("recommendations") is not None:
    st.warning("No recipes match your filters. Try increasing the calorie or time limits.")

else:
    st.info("👆 Click the button above to generate your personalized recommendations!")

# ============================================================
#                         SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 📍 Step 3/4")
    st.progress(0.75)

    st.markdown("---")
    sidebar_status()

    st.markdown("---")
    st.markdown("### 🧠 AI Method")
    st.markdown("""
    **Two-Tower Model:**
    - Deep learning architecture  
    - Learns both likes AND dislikes  
    - Projects 384D embeddings → 128D latent space  
    - Matches recipes by cosine similarity  
    """)
