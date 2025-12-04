"""
2_Swipe.py - Recipe Swiping Page (Tinder-style)
"""

import streamlit as st
import numpy as np

from utils.ui_components import inject_css
from utils.navigation import init_session_state, sidebar_status
from utils.data_loader import load_recipe_data, load_embeddings

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Swipe - Meal Planner AI",
    page_icon="❤️",
    layout="wide"
)

inject_css()
init_session_state()

# ========== DATA LOADING ==========
@st.cache_data
def get_swipe_recipes(n=100):
    """Load a sample of recipes for the swipe interface."""
    recipes_df, embeddings_df = load_recipe_data()
    
    if recipes_df is None:
        return None, None
    
    # Random sample
    sample_size = min(n, len(recipes_df))
    sample = recipes_df.sample(sample_size, random_state=42)
    
    return sample, embeddings_df

recipes_sample, embeddings_df = get_swipe_recipes()

# ========== HEADER ==========
st.markdown("# ❤️ Swipe Your Recipes")
st.markdown("Like the recipes you enjoy to train the AI!")

# Check data availability
if recipes_sample is None:
    st.error("❌ Unable to load recipes. Make sure `data/recipes_clean.csv` exists.")
    st.stop()

st.markdown("---")

# ========== PROGRESS ==========
likes_count = len(st.session_state.likes)
dislikes_count = len(st.session_state.dislikes)
total_swipes = likes_count + dislikes_count
min_likes_required = 5

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💚 Likes", likes_count)
with col2:
    st.metric("❌ Dislikes", dislikes_count)
with col3:
    st.metric("📊 Total", total_swipes)
with col4:
    progress = min(likes_count / min_likes_required, 1.0)
    st.metric("🎯 Goal", f"{likes_count}/{min_likes_required}")

st.progress(progress)

if likes_count >= min_likes_required:
    st.success(f"✅ Great! You liked {likes_count} recipes. You can now view your recommendations!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Continue Swiping", use_container_width=True):
            pass
    with col2:
        if st.button("🎯 View Recommendations", type="primary", use_container_width=True):
            st.switch_page("pages/3_Recommendations.py")

st.markdown("---")

# ========== SWIPE INTERFACE ==========
# IDs already seen
seen_ids = set(st.session_state.likes + st.session_state.dislikes)

# Filter unseen recipes
available_recipes = recipes_sample[~recipes_sample.index.isin(seen_ids)]

if len(available_recipes) == 0:
    st.warning("🎉 You have seen all available recipes!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart", use_container_width=True):
            st.session_state.likes = []
            st.session_state.dislikes = []
            st.rerun()
    with col2:
        if likes_count >= min_likes_required:
            if st.button("🎯 View Recommendations", type="primary", use_container_width=True):
                st.switch_page("pages/3_Recommendations.py")

else:
    # Take the next recipe
    current_recipe = available_recipes.iloc[0]
    current_id = available_recipes.index[0]
    
    # ========== SWIPE CARD ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="swipe-card">
            <div class="swipe-recipe-name">{current_recipe.get('name', 'Recipe')}</div>
            <div class="swipe-info">🔥 {int(current_recipe.get('calories', 0))} kcal</div>
            <div class="swipe-info">⏱️ {int(current_recipe.get('minutes', 0))} minutes</div>
            <div class="swipe-info">📝 {int(current_recipe.get('n_steps', 0))} steps</div>
            <div class="swipe-info">🥗 {int(current_recipe.get('n_ingredients', 0))} ingredients</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show ingredients if available
        if 'ingredients' in current_recipe.index and current_recipe['ingredients']:
            with st.expander("📋 View Ingredients"):
                ingredients = current_recipe['ingredients']
                if isinstance(ingredients, str):
                    try:
                        import ast
                        ingredients = ast.literal_eval(ingredients)
                    except:
                        ingredients = [ingredients]
                
                if isinstance(ingredients, list):
                    for ing in ingredients[:10]:
                        st.write(f"• {ing}")
                    if len(ingredients) > 10:
                        st.caption(f"... and {len(ingredients) - 10} more")
        
        # Nutrition info
        with st.expander("📊 Nutrition Information"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                protein = current_recipe.get('protein pvd', current_recipe.get('protein', 0))
                st.metric("🥩 Protein", f"{protein}%")
            with col_b:
                carbs = current_recipe.get('carbohydrates pvd', current_recipe.get('carbs', 0))
                st.metric("🍚 Carbs", f"{carbs}%")
            with col_c:
                fat = current_recipe.get('total fat pvd', current_recipe.get('fat', 0))
                st.metric("🥑 Fat", f"{fat}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Swipe buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("❌ Dislike", use_container_width=True, key="btn_dislike"):
                st.session_state.dislikes.append(int(current_id))
                st.rerun()
        
        with col_btn2:
            if st.button("⏭️ Skip", use_container_width=True, key="btn_skip"):
                st.session_state.dislikes.append(int(current_id))
                st.rerun()
        
        with col_btn3:
            if st.button("💚 Like", type="primary", use_container_width=True, key="btn_like"):
                st.session_state.likes.append(int(current_id))
                st.rerun()

st.markdown("---")

# ========== LIKED RECIPES ==========
if st.session_state.likes:
    with st.expander(f"💚 My Liked Recipes ({len(st.session_state.likes)})", expanded=False):
        liked_recipes = recipes_sample[recipes_sample.index.isin(st.session_state.likes)]
        
        for idx, recipe in liked_recipes.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"🍽️ **{recipe.get('name', 'Recipe')}**")
            with col2:
                st.caption(f"🔥 {int(recipe.get('calories', 0))} kcal")
            with col3:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.likes.remove(int(idx))
                    st.rerun()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📍 Step 2/4")
    st.progress(0.5)
    
    st.markdown("---")
    sidebar_status()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Like at least **5 recipes** for high-quality recommendations  
    - The more you like, the better the AI understands your taste  
    - Feel free to skip anything you don't like  
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.likes = []
        st.session_state.dislikes = []
        st.session_state.user_vec = None
        st.rerun()
