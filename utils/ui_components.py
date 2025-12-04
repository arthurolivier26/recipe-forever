"""
UI Components – Interface components and global CSS styles
"""

import streamlit as st


def inject_css():
    """Injects the global modern and clean CSS theme into Streamlit."""
    st.markdown("""
    <style>

        /* ========== GLOBAL LIGHT THEME ========== */
        .stApp {
            background-color: #f7f9fc;
            color: #1a1a1a;
        }

        /* ========== SIDEBAR ========== */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e6e6e6;
        }

        /* ========== HEADINGS ========== */
        h1, h2, h3, h4, h5 {
            color: #3b6ef8 !important;
            font-weight: 700 !important;
        }

        /* ========== BODY TEXT ========== */
        p, span, div {
            color: #333333;
        }

        /* ========== RECIPE CARDS ========== */
        .recipe-card {
            padding: 20px;
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid #e6e6e6;
            margin-bottom: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .recipe-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            border-color: #3b6ef8;
        }

        .recipe-title {
            color: #3b6ef8;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .recipe-meta {
            color: #777777;
            font-size: 13px;
        }

        /* ========== STAT BOX ========== */
        .stat-box {
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e6e6e6;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .stat-number {
            font-size: 28px;
            font-weight: bold;
            color: #3b6ef8;
        }

        .stat-label {
            color: #777777;
            font-size: 12px;
            margin-top: 5px;
        }

        /* ========== BUTTONS ========== */
        .stButton > button {
            background: linear-gradient(135deg, #4caf50 0%, #6ccf73 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.35);
        }

        /* ========== LIKE / DISLIKE / SKIP BUTTONS ========== */
        .like-btn button {
            background: linear-gradient(135deg, #4caf50, #6ccf73) !important;
        }

        .dislike-btn button {
            background: linear-gradient(135deg, #f44336, #ff7961) !important;
        }

        .skip-btn button {
            background: linear-gradient(135deg, #a7a7a7, #cfcfcf) !important;
        }

        /* ========== HERO SECTION ========== */
        .hero-title {
            font-size: 56px;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(135deg, #3b6ef8 0%, #9f77ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 20px 0;
            animation: fadeIn 1s;
        }

        .hero-subtitle {
            font-size: 20px;
            text-align: center;
            color: #777777;
            margin-bottom: 30px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ========== FEATURE CARDS ========== */
        .feature-card {
            background: #ffffff;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e6e6e6;
            margin: 15px 0;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }

        .feature-icon {
            font-size: 42px;
            margin-bottom: 10px;
        }

        /* ========== BADGES ========== */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px;
        }

        .badge-green {
            background: #e9f8ec;
            color: #4caf50;
            border: 1px solid #4caf50;
        }

        .badge-blue {
            background: #e8f0ff;
            color: #3b6ef8;
            border: 1px solid #3b6ef8;
        }

        .badge-orange {
            background: #fff3e3;
            color: #ff9800;
            border: 1px solid #ff9800;
        }

        .badge-red {
            background: #ffeaea;
            color: #e53935;
            border: 1px solid #e53935;
        }

        /* ========== TABLES ========== */
        .dataframe {
            background: white !important;
            border: 1px solid #e6e6e6 !important;
            border-radius: 8px;
        }

        .dataframe th {
            background: #f5f7ff !important;
            color: #3b6ef8 !important;
        }

        .dataframe td {
            color: #333333 !important;
            border-color: #e6e6e6 !important;
        }

        /* ========== PROGRESS BAR ========== */
        .progress-container {
            background: #e6e6e6;
            border-radius: 10px;
            padding: 3px;
            margin: 10px 0;
        }

        .progress-bar {
            background: linear-gradient(90deg, #4caf50 0%, #6ccf73 100%);
            height: 8px;
            border-radius: 8px;
            transition: width 0.5s;
        }

    </style>
    """, unsafe_allow_html=True)



def recipe_card(recipe: dict, show_score: bool = False):
    """
    Displays a stylized recipe card.

    Args:
        recipe: dict containing recipe information
        show_score: whether to display recommendation score
    """
    name = recipe.get('name', 'Recipe')
    calories = recipe.get('calories', 0)
    minutes = recipe.get('minutes', 0)
    score = recipe.get('score', 0)

    score_html = (
        f'<span class="badge badge-green">Score: {score:.2f}</span>'
        if show_score and score > 0 else ''
    )

    st.markdown(f"""
    <div class="recipe-card">
        <div class="recipe-title">{name}</div>
        <div class="recipe-meta">
            🔥 {calories} kcal &nbsp;|&nbsp; ⏱️ {minutes} min
            {score_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_box(value, label: str, emoji: str = "📊"):
    """Displays a stylized stat box."""
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{emoji} {value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def progress_indicator(current: int, total: int, label: str = ""):
    """Displays a graphical progress indicator."""
    percentage = (current / total * 100) if total > 0 else 0

    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">{label}</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%;"></div>
        </div>
        <div style="color: #58a6ff; font-size: 14px; text-align: right;">{current}/{total}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "blue"):
    """Creates a colored badge."""
    return f'<span class="badge badge-{color}">{text}</span>'


def meal_icon(meal_type: str) -> str:
    """Returns the emoji corresponding to a meal type."""
    icons = {
        'breakfast': '☕',
        'lunch': '🌞',
        'snack': '🍪',
        'dinner': '🌙'
    }
    return icons.get(meal_type.lower(), '🍽️')


def course_icon(course_type: str) -> str:
    """Returns the emoji for a course type."""
    icons = {
        'starter': '🥗',
        'main': '🥘',
        'dessert': '🍰'
    }
    return icons.get(course_type.lower(), '🍴')
