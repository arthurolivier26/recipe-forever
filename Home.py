"""
Home.py - Meal Planner AI Homepage
"""

import streamlit as st
import os

# ========== PAGE CONFIGURATION ==========
# Sets base display settings for the Streamlit app
st.set_page_config(
    page_title="Meal Planner AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== REQUIRED FILE CHECK ==========
# Paths to the folders that must contain trained models and datasets
MODELS_DIR = "./models"
DATA_DIR = "./data"

# Files required for the app to function properly
REQUIRED_FILES = {
    "models": [
        ("trained_user_towers.keras", "User Tower (Two-Tower)"),
        ("trained_recipe_towers.keras", "Recipe Tower (Two-Tower)"),
        ("TRANSFORMER_MODEL_PRETRAIN.keras", "Transformer Planning"),
    ],
    "data": [
        ("all_embeddings.csv", "Recipe + Token Embeddings"),
        ("recipes_clean.csv", "Recipe Database"),
        ("meals_category.csv", "Meal Categories"),
    ]
}


def check_required_files():
    """Check whether all mandatory model and data files exist."""
    missing = {"models": [], "data": []}
    
    # Check missing model files
    for filename, desc in REQUIRED_FILES["models"]:
        if not os.path.exists(os.path.join(MODELS_DIR, filename)):
            missing["models"].append((filename, desc))
    
    # Check missing dataset files
    for filename, desc in REQUIRED_FILES["data"]:
        if not os.path.exists(os.path.join(DATA_DIR, filename)):
            missing["data"].append((filename, desc))
    
    return missing


# Run the file check
missing = check_required_files()
has_missing = len(missing["models"]) > 0 or len(missing["data"]) > 0

# If required files are missing, display the installation help screen
if has_missing:
    st.error("## ⛔ Incomplete Installation")
    
    st.markdown("""
    **Meal Planner AI** requires trained models and dataset files to operate.  
    These files are produced by the Google Colab notebook.
    """)
    
    if missing["models"]:
        st.markdown("### 🧠 Missing Models (folder `models/`)")
        for filename, desc in missing["models"]:
            st.markdown(f"- ❌ `{filename}` — {desc}")
    
    if missing["data"]:
        st.markdown("### 📊 Missing Data Files (folder `data/`)")
        for filename, desc in missing["data"]:
            st.markdown(f"- ❌ `{filename}` — {desc}")
    
    st.markdown("""
    ---
    ### 📥 How to Get These Files

    1. Open **`REC_SYS_MAIN_PIPELINE.ipynb`** in Google Colab  
    2. Run all cells to train the models  
    3. Download the generated files from Google Drive  
    4. Place them inside the appropriate folders:
       - `models/` for `.keras` files  
       - `data/` for `.csv` files  
    5. Restart the application

    ---
    📁 **Expected Folder Structure:**
    ```
    MealPlanner/
    ├── models/
    │   ├── trained_user_towers.keras
    │   ├── trained_recipe_towers.keras
    │   └── TRANSFORMER_MODEL_PRETRAIN.keras
    ├── data/
    │   ├── all_embeddings.csv
    │   ├── recipes_clean.csv
    │   └── meals_category.csv
    └── ...
    ```
    """)
    
    st.stop()  # Prevent the rest of the app from running

# ========== IMPORT UTILITIES AFTER VALIDATION ==========
from utils.ui_components import inject_css
from utils.navigation import init_session_state, sidebar_status
from utils.data_loader import check_data_availability

# Initialize styles and session state
inject_css()
init_session_state()

# Check availability of files for status display
data_status = check_data_availability()

# ========== HERO SECTION ==========
st.markdown('<h1 class="hero-title">🍽️ Meal Planner AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Discover AI-powered personalized recipes</p>', unsafe_allow_html=True)

# Hero Image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image(
            "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=800&h=400&fit=crop",
            width='stretch'
        )
    except:
        st.info("🍽️ Welcome to Meal Planner AI")

st.markdown("<br>", unsafe_allow_html=True)

# ========== SYSTEM STATUS ==========
with st.expander("🔧 System Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Data Files**")
        st.write(f"{'✅' if data_status['recipes'] else '❌'} Recipes")
        st.write(f"{'✅' if data_status['embeddings'] else '❌'} Embeddings")
        st.write(f"{'✅' if data_status['categories'] else '❌'} Categories")
    
    with col2:
        st.markdown("**🧠 Machine Learning Models**")
        st.write(f"{'✅' if data_status['user_tower'] else '❌'} User Tower")
        st.write(f"{'✅' if data_status['recipe_tower'] else '❌'} Recipe Tower")
        st.write(f"{'✅' if data_status['transformer'] else '❌'} Transformer")
    
    with col3:
        st.markdown("**ℹ️ Status**")
        st.success("✅ System Ready!")

# ========== FEATURE OVERVIEW ==========
st.markdown("## ✨ How Does It Work?")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">👤</div>
        <h3>1. Profile</h3>
        <p>Enter your biometrics and nutrition goals</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">❤️</div>
        <h3>2. Swipe</h3>
        <p>Like or skip recipes Tinder-style</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h3>3. Recommendations</h3>
        <p>Receive personalized AI recipe suggestions</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📅</div>
        <h3>4. Weekly Planning</h3>
        <p>Generate your weekly meal plan</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ========== STATISTICS ==========
if data_status['recipes']:
    st.markdown("## 📊 Our Statistics")
    
    try:
        from utils.data_loader import load_recipes
        recipes_df = load_recipes()
        
        if recipes_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{len(recipes_df):,}</div>
                    <div class="stat-label">Recipes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_cal = int(recipes_df['calories'].mean()) if 'calories' in recipes_df.columns else 0
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{avg_cal}</div>
                    <div class="stat-label">Avg Calories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_time = int(recipes_df['minutes'].mean()) if 'minutes' in recipes_df.columns else 0
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{avg_time}</div>
                    <div class="stat-label">Avg Minutes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-number">🤖</div>
                    <div class="stat-label">AI Powered</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Unable to load statistics: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ========== CALL TO ACTION ==========
st.markdown("## 🚀 Get Started!")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h2>1️⃣</h2>
        <h3>Profile</h3>
        <p>Create your personalized nutrition profile</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📝 Create My Profile", key="btn_profile", args=None):
        st.switch_page("pages/1_Profile.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h2>2️⃣</h2>
        <h3>Swipe</h3>
        <p>Like the recipes you prefer</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("❤️ Start Swiping", key="btn_swipe"):
        st.switch_page("pages/2_Swipe.py")

with col3:
    st.markdown("""
    <div class="feature-card">
        <h2>3️⃣</h2>
        <h3>Recommendations</h3>
        <p>Discover your AI-generated recipes</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 View Recommendations", key="btn_reco"):
        st.switch_page("pages/3_Recommendations.py")

st.markdown("<br>", unsafe_allow_html=True)

st.info("👈 **Use the sidebar menu to navigate between pages!**")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 25px;'>
    <p><strong>Meal Planner AI</strong> • Powered by TensorFlow & Streamlit</p>
    <p>🔬 Two-Tower Model • 🤖 Transformer • 🎨 Modern Design</p>
    <p style='font-size: 11px; margin-top: 10px;'>© 2024 Meal Planner AI</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.info("Follow the steps in order:")
    
    st.markdown("""
    **📍 Pages:**
    1. 🏠 **Home** (You are here)
    2. 👤 **Profile** – Create your user profile
    3. ❤️ **Swipe** – Like recipes
    4. 🎯 **Recommendations** – Your AI suggestions
    5. 📅 **Weekly Planning** – Your weekly menu
    """)
    
    st.markdown("---")
    sidebar_status()
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🧠 **Two-Tower Model** (taste embeddings)
    - 🔄 **Transformer** (weekly meal planning)
    - 📊 **Sentence Transformers**
    - 🎨 **Streamlit** (UI framework)
    """)
