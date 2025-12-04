"""
4_Weekly_Planning.py — Weekly meal plan generation using the Transformer AI
NOTEBOOK VERSION — Uses Two-Tower (128D) + nutrition profile (132D) as input for the Transformer.
"""

import streamlit as st
import numpy as np
import os

# ============================================================
#                         PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Weekly Planning - Meal Planner AI",
    page_icon="📅",
    layout="wide"
)

# ============================================================
#                     MODEL VERIFICATION
# ============================================================

REQUIRED_MODELS = [
    "./models/trained_user_towers.keras",
    "./models/TRANSFORMER_MODEL_PRETRAIN.keras",
]

missing_models = [m for m in REQUIRED_MODELS if not os.path.exists(m)]
if missing_models:
    st.error("⛔ Missing models. Please return to the home page.")
    st.page_link("Home.py", label="🏠 Home", icon="🏠")
    st.stop()

from utils.ui_components import inject_css, meal_icon, course_icon
from utils.navigation import init_session_state, sidebar_status
from utils.data_loader import (
    load_recipe_data, load_embeddings, load_meals_category,
    load_user_tower, load_transformer_actor,
    load_all_embeddings_with_tokens
)
from utils.transformer_generator import generate_weekly_planning
from utils.user_profile import build_complete_user_profile

inject_css()
init_session_state()

# ============================================================
#                         LOAD DATA
# ============================================================

recipes_df, embeddings_df = load_recipe_data()
category_df = load_meals_category()

# Load models
user_tower = load_user_tower()
actor_model = load_transformer_actor()

if user_tower is None or actor_model is None:
    st.error("❌ Models could not be loaded.")
    st.stop()

# ============================================================
#                           HEADER
# ============================================================

st.markdown("# 📅 Your Weekly Meal Plan")
st.markdown(
    "Generate a fully personalized weekly menu using the **Transformer AI** from the notebook!"
)

if recipes_df is None:
    st.error("❌ Recipe data is not available.")
    st.stop()

# Require at least 3 liked recipes
if len(st.session_state.likes) < 3:
    st.warning("⚠️ Like at least 3 recipes to generate a personalized weekly plan.")
    if st.button("❤️ Start swiping"):
        st.switch_page("pages/2_Swipe.py")
    st.stop()

st.markdown("---")

# ============================================================
#                      PLANNING OPTIONS
# ============================================================

st.markdown("### ⚙️ Planning Options")

col1, col2 = st.columns(2)

with col1:
    n_days = st.slider("Number of days", 1, 14, 7)

with col2:
    if st.session_state.get("target_calories"):
        target_cal = st.session_state.target_calories
        st.metric("🎯 Goal (from profile)", f"{int(target_cal)} kcal/day")
    else:
        target_cal = st.number_input("Calories/day (if not in profile)", 1200, 4000, 2000)

st.markdown("---")

# ============================================================
#                  WEEKLY PLAN GENERATION
# ============================================================

if st.button("🚀 Generate my weekly plan", type="primary", use_container_width=True):

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1 — Validate profile
        status_text.text("👤 Checking user profile...")
        progress_bar.progress(10)

        profile = st.session_state.get("profile", {})
        if not profile:
            st.warning("⚠️ Incomplete profile. Using default values.")
            profile = {
                "weight": 70,
                "height": 170,
                "age": 30,
                "gender": "male",
                "activity": "active",
                "objective": "maintenance",
            }

        # Step 2 — Build the full 132D user vector (Two-Tower + nutrition)
        status_text.text("🧠 Computing complete user profile...")
        progress_bar.progress(30)

        complete_profile = build_complete_user_profile(
            liked_ids=st.session_state.likes,
            disliked_ids=st.session_state.dislikes,
            embeddings_df=embeddings_df,
            user_tower_model=user_tower,
            weight=profile.get("weight", 70),
            height=profile.get("height", 170),
            age=profile.get("age", 30),
            gender=profile.get("gender", "male"),
            activity=profile.get("activity", "active"),
            objective=profile.get("objective", "maintenance"),
        )

        user_vec_132 = complete_profile["user_vec_132"]
        target_cal = complete_profile["target_calories"]

        # Update session
        st.session_state.user_vec = user_vec_132
        st.session_state.target_calories = target_cal
        st.session_state.complete_profile = complete_profile

        progress_bar.progress(50)

        # Step 3 — Load embeddings including transformer special tokens
        status_text.text("📊 Loading embeddings...")
        all_embeddings = load_all_embeddings_with_tokens()

        if all_embeddings is None:
            st.error("❌ Could not load all_embeddings.csv")
            st.stop()

        progress_bar.progress(60)

        # Step 4 — Transformer generation
        status_text.text("🤖 Generating with Transformer AI...")

        planning = generate_weekly_planning(
            user_vec=user_vec_132,
            recipes_df=recipes_df,
            embeddings_df=embeddings_df,
            category_df=category_df,
            actor_model=actor_model,
            all_embeddings_df=all_embeddings,
            token_registry=None,
            days=n_days,
            use_transformer=True,
            target_calories=target_cal,
        )

        progress_bar.progress(90)

        # Step 5 — Finalization
        status_text.text("✅ Finalizing...")

        st.session_state.planning = planning
        st.session_state.planning_method = "Transformer (Notebook)"

        progress_bar.progress(100)
        status_text.empty()

        st.success("✅ Weekly plan generated using **Transformer AI**!")
        st.balloons()

        # Show nutritional profile breakdown
        with st.expander("📊 Your nutritional profile"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🔥 BMR", f"{int(complete_profile['bmr'])} kcal")
                st.metric("⚡ TDEE", f"{int(complete_profile['tdee'])} kcal")

            with col2:
                st.metric("🎯 Goal", f"{int(target_cal)} kcal/day")
                st.metric("💪 Proteins", f"{complete_profile['target_protein_g']} g/day")

            with col3:
                st.metric("📐 BMI", complete_profile["bmi"])
                macros = complete_profile["macros"]
                st.caption(
                    f"Macros: {macros['protein_pct']}% P / "
                    f"{macros['carbs_pct']}% C / {macros['fat_pct']}% F"
                )

    except Exception as e:
        st.error(f"❌ Error during generation: {e}")
        import traceback
        with st.expander("🐛 Error details"):
            st.code(traceback.format_exc())


# ============================================================
#                  DISPLAYING THE WEEKLY PLAN
# ============================================================

if st.session_state.get("planning") is not None:

    planning = st.session_state.planning
    method = st.session_state.get("planning_method", "Default")

    st.markdown("---")

    # Global statistics
    stats = planning.get("stats", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📅 Days", stats.get("num_days", 0))

    with col2:
        st.metric("🍽️ Recipes", stats.get("total_recipes", 0))

    with col3:
        st.metric("🔥 kcal/day", stats.get("avg_calories_per_day", 0))

    with col4:
        st.metric("🤖 Method", method)

    st.markdown("---")

    # Daily breakdown
    for day_data in planning.get("days", []):
        day_num = day_data.get("day", 0)

        st.markdown(f"### 📆 Day {day_num}")

        meals = day_data.get("meals", [])

        if not meals:
            st.caption("No meals for this day.")
            continue

        cols = st.columns(len(meals))

        for i, meal in enumerate(meals):
            meal_type = meal.get("type", "meal")
            courses = meal.get("courses", [])

            with cols[i]:
                icon = meal_icon(meal_type)
                label = {
                    "breakfast": "Breakfast",
                    "lunch": "Lunch",
                    "snack": "Snack",
                    "dinner": "Dinner",
                }.get(meal_type, meal_type.title())

                st.markdown(f"**{icon} {label}**")

                total_cal = 0

                for course in courses:
                    course_type = course.get("type", "main")
                    recipe = course.get("recipe", {})

                    c_icon = course_icon(course_type)
                    name = recipe.get("name", "Recipe")
                    if len(name) > 35:
                        name = name[:32] + "..."

                    calories = recipe.get("calories", 0)
                    minutes = recipe.get("minutes", 0)
                    taste_score = recipe.get("taste_score", 0)

                    total_cal += calories

                    taste_emoji = (
                        "💚" if taste_score > 0.8 else
                        "💛" if taste_score > 0.6 else
                        "💙"
                    )

                    st.markdown(f"""
                    <div style="padding: 8px; margin: 4px 0; background: #f8f9fa;
                                border-radius: 6px; border-left: 3px solid #4caf50;">
                        <div style="font-weight: 600; color: #333;">
                            {c_icon} {name}
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 4px;">
                            🔥 {calories} kcal &nbsp;|&nbsp; ⏱️ {minutes} min
                            {' &nbsp;|&nbsp; ' + taste_emoji + f' {taste_score:.2f}' if taste_score > 0 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"**Total: {total_cal} kcal**")

        # Daily stats if available
        if "stats" in day_data:
            day_stats = day_data["stats"]
            with st.expander(f"📊 Day {day_num} statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Calories", f"{int(day_stats.get('total_kcal', 0))} kcal")
                with col2:
                    st.metric("Proteins", f"{day_stats.get('total_pvd', 0):.0f}% DV")
                with col3:
                    delta = day_stats.get("delta_target_kcal", 0)
                    st.metric("Calorie gap", f"{delta:+.0f} kcal")

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.planning = None
            st.rerun()

    with col2:
        if st.button("📥 Export (JSON)", use_container_width=True):
            import json
            planning_json = json.dumps(planning, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Download",
                data=planning_json,
                file_name="meal_planning.json",
                mime="application/json",
            )

    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("Home.py")

    # Global nutrition audit
    with st.expander("📊 Global nutrition details"):
        if "global_audit" in planning or "stats" in planning:
            audit = planning.get("global_audit", planning.get("stats", {}))
            st.json(audit)
        else:
            st.info("No audit data available.")

else:
    st.info("👆 Adjust the settings above and click **Generate** to create your weekly plan!")

    st.markdown("### 📋 What you will get")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Daily structure:**
        - ☕ Breakfast
        - 🌞 Lunch (starter + main + dessert)
        - 🍪 Snack (optional)
        - 🌙 Dinner (starter + main + dessert)
        """)

    with col2:
        st.markdown("""
        **Personalization includes:**
        - Based on your likes/dislikes  
        - Nutrition-adapted  
        - Recipe variety  
        - Macro balancing  
        - Dynamic calorie regulation  
        """)

# ============================================================
#                         SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 📍 Step 4/4")
    st.progress(1.0)

    st.markdown("---")
    sidebar_status()

    st.markdown("---")
    st.markdown("### 🧠 Transformer System")

    actor = load_transformer_actor()

    if actor is not None:
        st.success("✅ Transformer model available")
        st.caption("✨ **Notebook features:**")
        st.caption("- VocabManager token system")
        st.caption("- 132D preference + nutrition vector")
        st.caption("- Multi-criteria reranking")
        st.caption("- Dynamic regulation (failsafes & corrections)")
    else:
        st.warning("⚠️ Transformer model unavailable")
        st.caption("Check file: TRANSFORMER_MODEL_PRETRAIN.keras")

    st.markdown("---")
    st.markdown("### 📖 Features")
    st.caption("""
    **Smart generation:**
    - Adaptive meal skeleton  
    - Strict category filtering  
    - Taste score + protein + calorie budgeting  
    - Recipe diversity  
    - Full nutrition audit  
    """)
