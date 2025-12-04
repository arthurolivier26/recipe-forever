"""
1_Profile.py — User profile creation page for Meal Planner AI
"""

import streamlit as st
from utils.ui_components import inject_css
from utils.navigation import init_session_state, sidebar_status
from utils.user_profile import (
    calculate_bmr, calculate_tdee, objective_adjust,
    calculate_macros, calculate_bmi, get_bmi_category
)

# ============================================================
#                         PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Profile - Meal Planner AI",
    page_icon="👤",
    layout="wide"
)

inject_css()
init_session_state()

# ============================================================
#                           HEADER
# ============================================================
st.markdown("# 👤 Your Nutrition Profile")
st.markdown("Fill in your personal information to calculate your personalized calorie needs.")

st.markdown("---")

# ============================================================
#                         FORM SECTION
# ============================================================
col1, col2 = st.columns(2)

# ------------------------- Personal Info -------------------------
with col1:
    st.markdown("### 📋 Personal Information")

    gender = st.selectbox(
        "Gender",
        options=["male", "female"],
        format_func=lambda x: "👨 Male" if x == "male" else "👩 Female",
        index=0 if st.session_state.profile.get("gender") != "female" else 1
    )

    age = st.slider(
        "Age",
        min_value=16,
        max_value=90,
        value=st.session_state.profile.get("age", 30)
    )

    height = st.slider(
        "Height (cm)",
        min_value=140,
        max_value=220,
        value=st.session_state.profile.get("height", 175)
    )

    weight = st.slider(
        "Weight (kg)",
        min_value=40,
        max_value=200,
        value=st.session_state.profile.get("weight", 70)
    )

# ------------------------- Activity & Goal -------------------------
with col2:
    st.markdown("### 🏃 Activity & Goal")

    activity = st.selectbox(
        "Daily activity level",
        options=["sedentary", "active", "very_active"],
        format_func=lambda x: {
            "sedentary": "🪑 Sedentary (desk job, low movement)",
            "active": "🚶 Active (regular walking, standing)",
            "very_active": "🏃 Very active (manual work, daily sport)"
        }[x],
        index=["sedentary", "active", "very_active"].index(
            st.session_state.profile.get("activity", "sedentary")
        )
    )

    sport_hours = st.slider(
        "Hours of exercise per week",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state.profile.get("sport_hours", 3.0)),
        step=0.5
    )

    objective = st.selectbox(
        "Goal",
        options=["maintenance", "weight_loss", "muscle_gain"],
        format_func=lambda x: {
            "maintenance": "⚖️ Maintain weight",
            "weight_loss": "📉 Lose weight",
            "muscle_gain": "💪 Gain muscle"
        }[x],
        index=["maintenance", "weight_loss", "muscle_gain"].index(
            st.session_state.profile.get("objective", "maintenance")
        )
    )

st.markdown("---")

# ============================================================
#                     CALCULATION BUTTON
# ============================================================
if st.button("✨ Calculate my needs", type="primary", use_container_width=True):

    # Compute caloric needs and nutrition profile
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(weight, height, age, gender, activity, sport_hours)
    target_calories = objective_adjust(tdee, objective)
    macros = calculate_macros(target_calories, objective, weight)
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)

    # Save to session state
    st.session_state.profile = {
        "gender": gender,
        "age": age,
        "height": height,
        "weight": weight,
        "activity": activity,
        "sport_hours": sport_hours,
        "objective": objective,
        "bmr": bmr,
        "tdee": tdee,
        "bmi": bmi,

        # Needed for constructing the 132D user vector
        "protein_g": macros["protein_g"],
        "fat_g": macros["fat_g"],
        "carbs_g": macros["carbs_g"],

        # Critical: include macro percentages for user_vec_132
        "protein_pct": macros["protein_pct"],
        "fat_pct": macros["fat_pct"],
        "carbs_pct": macros["carbs_pct"]
    }

    st.session_state.target_calories = int(target_calories)

    st.success("✅ Profile calculated successfully!")

    # ============================================================
    #                         RESULTS DISPLAY
    # ============================================================
    st.markdown("## 📊 Your Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 Basal Metabolic Rate", f"{int(bmr)} kcal")
        st.caption("Calories burned at rest")

    with col2:
        st.metric("⚡ Total Daily Energy Expenditure", f"{int(tdee)} kcal")
        st.caption("Based on your activity level")

    with col3:
        st.metric("🎯 Calorie target", f"{int(target_calories)} kcal")
        st.caption(f"For: {objective.replace('_', ' ')}")

    with col4:
        st.metric("📏 BMI", f"{bmi}")
        st.caption(bmi_category)

    st.markdown("---")

    # --------------------------- Macros ---------------------------
    st.markdown("### 🥗 Macronutrient Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">🥩 {macros['protein_g']}g</div>
            <div class="stat-label">Protein ({macros['protein_pct']}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">🍚 {macros['carbs_g']}g</div>
            <div class="stat-label">Carbs ({macros['carbs_pct']}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">🥑 {macros['fat_g']}g</div>
            <div class="stat-label">Fats ({macros['fat_pct']}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Next step button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Continue to swiping", type="primary", use_container_width=True):
            st.switch_page("pages/2_Swipe.py")

# ============================================================
#              DISPLAY EXISTING PROFILE IF ALREADY SAVED
# ============================================================
elif st.session_state.profile:
    st.markdown("## 📋 Your Current Profile")

    profile = st.session_state.profile

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        st.write(f"Gender: {'👨 Male' if profile['gender'] == 'male' else '👩 Female'}")
        st.write(f"Age: {profile['age']} years")
        st.write(f"Height: {profile['height']} cm")
        st.write(f"Weight: {profile['weight']} kg")

    with col2:
        st.markdown("**🏃 Activity**")
        activity_labels = {
            "sedentary": "Sedentary",
            "active": "Active",
            "very_active": "Very active"
        }
        st.write(f"Activity level: {activity_labels.get(profile['activity'], profile['activity'])}")
        st.write(f"Sport: {profile['sport_hours']}h/week")

        objective_labels = {
            "maintenance": "Maintain weight",
            "weight_loss": "Weight loss",
            "muscle_gain": "Muscle gain"
        }
        st.write(f"Goal: {objective_labels.get(profile['objective'], profile['objective'])}")

    with col3:
        st.markdown("**📊 Results**")
        st.write(f"BMR: {int(profile.get('bmr', 0))} kcal")
        st.write(f"TDEE: {int(profile.get('tdee', 0))} kcal")
        st.write(f"🎯 Calorie target: **{st.session_state.target_calories} kcal**")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Edit my profile", use_container_width=True):
            st.session_state.profile = {}
            st.rerun()
    with col2:
        if st.button("➡️ Continue to swiping", type="primary", use_container_width=True):
            st.switch_page("pages/2_Swipe.py")

# ============================================================
#                           SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📍 Step 1/4")
    st.progress(0.25)

    st.markdown("---")
    sidebar_status()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Be honest about your activity level  
    - Exercise hours are in addition to daily activity  
    - Goals adjust calories by ±300–500 kcal  
    """)
