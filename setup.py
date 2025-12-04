"""
setup.py - Script d'initialisation de Meal Planner AI
Vérifie et crée la structure nécessaire
"""

import os
import sys

def create_directories():
    """Crée la structure de dossiers"""
    dirs = ["data", "models", "utils", "pages"]
    
    print("📁 Création des dossiers...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   ✅ {d}/")

def check_files():
    """Vérifie les fichiers requis"""
    
    files = {
        # Obligatoires
        "data/recipes_clean.csv": ("OBLIGATOIRE", "Données des recettes"),
        "data/all_embeddings.csv": ("OBLIGATOIRE", "Embeddings recettes + tokens"),
        "data/meals_category.csv": ("RECOMMANDÉ", "Catégories de repas"),
        
        # Modèles (optionnels mais recommandés)
        "models/trained_user_towers.keras": ("OBLIGATOIRE", "Modèle User Tower (Two-Tower)"),
        "models/trained_recipe_towers.keras": ("OBLIGATOIRE", "Modèle Recipe Tower (Two-Tower)"),
        "models/TRANSFORMER_MODEL_PRETRAIN.keras": ("OBLIGATOIRE", "Modèle Transformer (Planning)"),
    }
    
    print("\n📋 Vérification des fichiers...")
    print("=" * 60)
    
    missing_required = []
    missing_optional = []
    
    for path, (level, desc) in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {path:<40} ({size:.1f} MB)")
        else:
            print(f"❌ {path:<40} [{level}]")
            if level == "OBLIGATOIRE":
                missing_required.append((path, desc))
            else:
                missing_optional.append((path, desc))
    
    print("=" * 60)
    
    if missing_required:
        print("\n⚠️  FICHIERS OBLIGATOIRES MANQUANTS:")
        for path, desc in missing_required:
            print(f"   - {path}")
            print(f"     → {desc}")
        print("\n   L'application ne fonctionnera PAS sans ces fichiers!")
    
    if missing_optional:
        print("\n💡 FICHIERS OPTIONNELS MANQUANTS:")
        for path, desc in missing_optional:
            print(f"   - {path}")
            print(f"     → {desc}")
        print("\n   L'app fonctionnera avec des fonctionnalités réduites.")
    
    if not missing_required and not missing_optional:
        print("\n🎉 TOUS LES FICHIERS SONT PRÉSENTS!")
    
    return len(missing_required) == 0

def show_export_instructions():
    """Affiche les instructions d'export depuis le notebook"""
    
    print("\n" + "=" * 60)
    print("📚 COMMENT EXPORTER DEPUIS TON NOTEBOOK")
    print("=" * 60)
    
    print("""
📍 ÉTAPE 1: Exporter les Embeddings (all_embeddings.csv)
─────────────────────────────────────────────────────────
Dans ton notebook, après la création de ALL_EMBEDDINGS:

```python
# Export all_embeddings (recettes + tokens)
ALL_EMBEDDINGS.to_csv("all_embeddings.csv")
print(f"✅ Exporté: {len(ALL_EMBEDDINGS)} embeddings")
```

Puis copie le fichier dans: data/all_embeddings.csv


📍 ÉTAPE 2: Exporter les Recettes (recipes_clean.csv)
─────────────────────────────────────────────────────────
Déjà dans ton Google Drive:
/content/drive/MyDrive/REC_SYS_PROJECT/DATA FINALE/recipes_clean.csv

Copie-le dans: data/recipes_clean.csv


📍 ÉTAPE 3: Exporter les Catégories (meals_category.csv)
─────────────────────────────────────────────────────────
Déjà dans ton Google Drive:
/content/drive/MyDrive/REC_SYS_PROJECT/DATA FINALE/meals_category.csv

Copie-le dans: data/meals_category.csv


📍 ÉTAPE 4: Exporter les Modèles Two-Tower (optionnel)
─────────────────────────────────────────────────────────
Dans ton notebook, après l'entraînement Two-Tower:

```python
# Export des modèles Two-Tower
user_tower_layer.save("user_tower.keras")
recipe_tower_layer.save("recipe_tower.keras")
print("✅ Modèles Two-Tower exportés")
```

Copie dans: models/user_tower.keras et models/recipe_tower.keras


📍 ÉTAPE 5: Exporter le Transformer (optionnel)
─────────────────────────────────────────────────────────
Dans ton notebook, après l'entraînement:

```python
# Export du Transformer Actor
actor.save("actor_final.keras")
print("✅ Transformer exporté")
```

Copie dans: models/actor_final.keras
""")

def main():
    print("🚀 SETUP MEAL PLANNER AI")
    print("=" * 60)
    
    # 1. Créer les dossiers
    create_directories()
    
    # 2. Vérifier les fichiers
    all_ok = check_files()
    
    # 3. Instructions si fichiers manquants
    if not all_ok:
        show_export_instructions()
    
    # 4. Résumé
    print("\n" + "=" * 60)
    print("📌 PROCHAINES ÉTAPES")
    print("=" * 60)
    
    if all_ok:
        print("""
✅ Tout est prêt !

Pour lancer l'application:

   Option 1 - Script batch (Windows):
   > start.bat

   Option 2 - Manuellement:
   Terminal 1: python main.py
   Terminal 2: streamlit run Home.py

📊 URLs:
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
""")
    else:
        print("""
⚠️  Des fichiers sont manquants.

1. Exporte les fichiers depuis ton notebook (voir instructions ci-dessus)
2. Place-les dans les bons dossiers
3. Relance: python setup.py
4. Puis lance l'application
""")

if __name__ == "__main__":
    main()
