"""
Navigation — Utility helpers for page navigation in Streamlit apps.
"""

import streamlit as st


# ============================================================
#                  NAVIGATION HELPERS
# ============================================================

def is_new_navigation_available():
    """
    Check whether Streamlit's modern navigation API (st.switch_page)
    is available. It requires Streamlit >= 1.30.
    """
    return hasattr(st, "switch_page")


def navigate_to(page_path: str):
    """
    Navigate to another Streamlit page.

    Args:
        page_path: Path to the target page file
                   e.g. "pages/1_Profile.py".
    """
    if is_new_navigation_available():
        try:
            st.switch_page(page_path)
        except Exception as e:
            st.error(f"Navigation error: {e}")
            st.info(f"👈 Use the sidebar menu to navigate to {page_path}")
    else:
        # Fallback for older Streamlit versions:
        # Store the target and let the sidebar or app logic handle it.
        st.session_state["_navigation_target"] = page_path
        st.info(f"👈 Use the sidebar menu to go to: {page_path}")


# ============================================================
#               SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """
    Initialize all required session state variables with default values
    if they do not already exist.
    """
    defaults = {
        "profile": {},
        "likes": [],
        "dislikes": [],
        "user_vec": None,
        "target_calories": None,
        "swipe_index": 0,
        "recommendations": None,
        "planning": None
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ============================================================
#                        PROGRESS TRACKER
# ============================================================

def get_progress():
    """
    Compute the user’s progress through the application workflow.

    Returns:
        progress (int): percentage completed (0‒100)
        steps (list): list of tuples (label, completed_bool)
    """
    progress = 0
    steps = []

    # Step 1 — Profile created
    if st.session_state.get("profile"):
        progress += 25
        steps.append(("✅ Profile created", True))
    else:
        steps.append(("⏳ Create profile", False))

    # Step 2 — Enough likes (minimum 5)
    likes_count = len(st.session_state.get("likes", []))
    if likes_count >= 5:
        progress += 25
        steps.append((f"✅ {likes_count} likes", True))
    else:
        steps.append((f"⏳ {likes_count}/5 likes", False))

    # Step 3 — User vector computed
    if st.session_state.get("user_vec") is not None:
        progress += 25
        steps.append(("✅ AI profile generated", True))
    else:
        steps.append(("⏳ AI profile pending", False))

    # Step 4 — Recommendations displayed
    if st.session_state.get("recommendations") is not None:
        progress += 25
        steps.append(("✅ Recommendations ready", True))
    else:
        steps.append(("⏳ Recommendations", False))

    return progress, steps


# ============================================================
#                   SIDEBAR STATUS DISPLAY
# ============================================================

def sidebar_status():
    """
    Display the user’s progress and swipe statistics inside the Streamlit sidebar.
    """

    progress, steps = get_progress()

    st.sidebar.markdown("### 📈 Progress")
    st.sidebar.progress(progress / 100)
    st.sidebar.caption(f"{progress}% completed")

    # Display each step status
    for step_text, completed in steps:
        if completed:
            st.sidebar.success(step_text)
        else:
            st.sidebar.info(step_text)

    # Display calorie target if available
    if st.session_state.get("target_calories"):
        st.sidebar.markdown("---")
        st.sidebar.metric("🎯 Goal", f"{st.session_state.target_calories} kcal")

    # Like / Dislike statistics
    likes = len(st.session_state.get("likes", []))
    dislikes = len(st.session_state.get("dislikes", []))

    if likes > 0 or dislikes > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Swipe Stats")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("💚 Likes", likes)
        col2.metric("❌ Dislikes", dislikes)
