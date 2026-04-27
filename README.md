# Détection de Fraude Bancaire — TP_Decision_Tree

Ce dépôt contient un notebook de TP dédié à la détection de fraude sur des transactions bancaires. Le notebook guide l'étudiant pas à pas depuis l'exploration des données jusqu'à l'évaluation d'un classifieur supervisé.

## Fichiers
- [TP_classification_Fraud.ipynb](TP_classification_Fraud.ipynb) — Notebook principal : EDA, nettoyage, feature engineering, pipeline, recherche d'hyperparamètres et évaluation.
- [Fraud Detection Dataset.csv](Fraud%20Detection%20Dataset.csv) — Jeu de données (à placer dans le même dossier que le notebook).

## Objectif du projet
- Construire un pipeline de classification pour détecter les transactions frauduleuses.
- Présenter une démarche reproductible et pédagogique : analyser les données, préparer les features, entraîner un modèle, et interpréter les résultats.

## Ce que nous avons fait (résumé)
1. Chargement et inspection du dataset
	- Vérification des dimensions, types, statistiques et doublons.
2. Analyse exploratoire (EDA)
	- Visualisation des distributions, étude des valeurs manquantes et de la distribution de la cible.
3. Nettoyage et imputation
	- Remplacement des valeurs manquantes sur les colonnes catégorielles par `Unknown` et imputation des numériques par la médiane.
4. Feature engineering
	- Création de plages horaires (`Hour_of_Day`), transformation logarithmique du montant (`Log_Transaction_Amount`) et catégorisation par quantiles (`Amount_Category`).
5. Préparation des données et pipeline
	- Séparation train/test stratifiée.
	- Construction d'un préprocesseur (`StandardScaler` pour numériques, `OneHotEncoder` pour catégoriques) intégré dans un `Pipeline`.
6. Baseline
	- Évaluation d'un classifieur naïf (`DummyClassifier`) comme référence.
7. Modélisation et hyperparamétrage
	- Entraînement d'un `DecisionTreeClassifier` avec `GridSearchCV` (optimisation sur le F1-score) et validation croisée stratifiée.
8. Évaluation
	- Calcul des métriques (Accuracy, Precision, Recall, F1, ROC-AUC), matrice de confusion, courbes ROC et Precision-Recall, importance des features et analyse de sensibilité au seuil.

## Résultats clés (à lire dans le notebook)
- Rapport de classification et matrice de confusion sur l'ensemble de test.
- Liste et visualisation des features les plus importantes selon le Decision Tree.
- Analyse de l'impact du seuil de décision sur precision/recall/F1 — utile pour adapter le modèle en production selon la tolérance au risque.

## Environnement et dépendances
Recommandé : créer un environnement virtuel puis installer les paquets nécessaires.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

Si vous préférez un fichier de dépendances, je peux générer un `requirements.txt`.

## Exécution
1. Démarrer Jupyter Lab / Notebook :

```bash
jupyter lab
# ou
jupyter notebook
```

2. Ouvrir [TP_classification_Fraud.ipynb](TP_classification_Fraud.ipynb) et exécuter les cellules dans l'ordre.

Remarque : certaines cellules dépendent des résultats précédents (préprocesseur, modèles entraînés). Si vous voulez exécuter partiellement, adaptez les cellules ou rechargez les objets nécessaires.

## Bonnes pratiques et prochaines étapes suggérées
- Tester d'autres modèles : `RandomForest`, `XGBoost`, `LightGBM`.
- Gérer le déséquilibre : undersampling, oversampling (SMOTE), ou ajuster `class_weight`.
- Pipeline de production : sérialiser le préprocesseur + modèle (`joblib`), ajouter tests unitaires simples, et construire un script d'inférence minimal.
- Monitoring en production : logs, dérive des features, recalibrage périodique.

## Auteur / Contexte
Notebook conçu pour un TP étudiant : code et commentaires ont été simplifiés pour la lisibilité et l'apprentissage.

---

