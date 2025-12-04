# Meal Planner AI – Technical Documentation

This repository contains a full Meal Planning Recommender System combining a Two-Tower architecture, a Transformer-based sequence generator, and a Streamlit interface for user interaction.

---

## 1. Project Overview

Meal Planner AI generates personalized weekly meal plans using:

- User preferences collected via a swipe interface
- Nutritional profile modeling (calories, macros, BMI, activity)
- A Two-Tower recommender system producing 128-dimensional taste embeddings
- A Transformer-based actor model generating structured meal sequences
- Recipe reranking using KNN, nutrition constraints, and user similarity

---

## 2. Architecture Diagram

The system architecture is shown below:

┌─────────────────────────┐
│        Streamlit UI     │
│  - Home page            │
│  - User setup           │
│  - Swipe interface      │
│  - Recommendations       │
│  - Weekly planner        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│         Backend         │
│         FastAPI         │
│ Exposes inference APIs  │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│               AI Models                  │
│                                          │
│ Two-Tower Recommender                    │
│ - User Tower (preference encoding)       │
│ - Recipe Tower (recipe embedding)        │
│                                          │
│ Transformer Generator                    │
│ - Structural token prediction            │
│ - Recipe embedding prediction            │
│ - Nutrition vector prediction            │
│                                          │
│ Multi-Criteria Reranking Module          │
│ - Taste similarity                       │
│ - Calorie budget penalty                 │
│ - Protein density penalty                │
└──────────────────────────────────────────┘


## 3. System Components

### 3.1 Streamlit UI  
Handles all user-facing features:
- Swipe interface (similar to Tinder)
- Recommendation browsing
- Weekly meal planning
- Sidebar navigation and progress tracking

### 3.2 Swipe Engine  
Collects explicit feedback:
- Like / dislike interactions  
- Builds the user’s preference vector using the Two-Tower model  

### 3.3 User Profile Builder  
Computes:
- BMR, TDEE  
- Calorie goals  
- Macro targets  
- BMI categories  
- Normalized 132-dimensional metabolic vectors  

### 3.4 Two-Tower Model  
Outputs:
- User embedding (128D taste vector)
- Recipe embedding matrix used for KNN and similarity search

### 3.5 Transformer Generator  
Generates structured sequences of meals:
- Breakfast, Lunch, Snack, Dinner
- Starter → Main dish → Dessert logic  
- Budget-aware caloric constraints  
- Protein-density preservation  
- Optional meal skipping or addition when needed  

### 3.6 Recipe Tower  
Projects recipe embeddings into latent space for taste scoring.

---

## 4. Data Pipeline

1. Load recipe metadata and nutritional values  
2. Load embedding matrices for recipes  
3. Build user taste vector via likes/dislikes  
4. Build metabolism-derived vector (calorie + macro normalization)  
5. Combine into full 132D user vector  
6. Feed into Transformer to generate meal sequence  
7. For every `<RECIPE_SLOT>`:  
   - Predict target embedding + nutrition  
   - Retrieve candidates via KNN  
   - Rerank using multi-criteria scoring  

---

## 5. File Structure

```
/MealPlannerAI
│
├── main.py                    # FastAPI backend
├── Home.py                    # Streamlit homepage
│
├── utils/
│   ├── user_profile.py        # Calorie, macros, BMI, user vector
│   ├── transformer_generator.py # Weekly plan generation logic
│   ├── recommender.py         # Two-tower recommendation
│   ├── navigation.py          # Session state
│   ├── ui_components.py       # CSS and UI widgets
│   ├── token_registry.py      # Structural tokens
│   ├── data_loader.py         # Load CSVs, embeddings, models
│
├── pages/
│   ├── 1_Profile.py           # User setup
│   ├── 2_Swipe.py             # Tinder-like interface
│   ├── 3_Recommendations.py   # Recommendations
│   └── 4_Planner.py           # Weekly planner
│
├── models/                    # Saved TensorFlow models
├── data/                      # Recipes, embeddings, metadata
├── start.bat                  # Launcher script (Windows)
└── setup.py                   # File integrity checks
requirement.txt

```

---

## 6. Installation

```
git clone <repository-url>
cd MealPlannerAI
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python setup.py
```

---

## 7. Running the Application

Option 1 (recommended):
```
start.bat
```

Option 2 (manual launch):
```
python main.py
streamlit run Home.py
```

---

## 8. Algorithms & Metrics

### Two-Tower Similarity  
Cosine similarity in 128D embedding space.

### Transformer Prediction  
Outputs three tensors:
- Next structural token
- Target recipe embedding
- Target nutrition vector (kcal + protein-density)

### Reranking Formula  
For each candidate recipe:
```
score = 5 * taste_similarity
      - 2 * calorie_penalty
      - 1 * protein_density_penalty
```

### Budget Regulation  
Cuts optional dishes if exceeding threshold.  
Adds snack when below required calories.

---

## 9. Troubleshooting

| Issue | Cause | Solution |
|-------|--------|-----------|
| Missing models | Incorrect export paths | Check models/ directory |
| recipes_clean.csv not found | Wrong dataset location | Place file in data/ folder |
| Embedding size mismatch | Wrong version of embedding file | Re-export from notebook |


