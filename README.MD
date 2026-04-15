# Concevoir une IA simple

Ce projet individuel présente un workflow de régression prédictive en IA.

## Objectif

- Construire un notebook propre et reproductible
- Préparer les données de manière rigoureuse
- Tester plusieurs modèles de régression
- Comparer les performances et expliquer les résultats

## Livrables

- Notebook Jupyter documenté avec code, graphiques et interprétations
- README synthétique expliquant la méthodologie et les choix techniques
- Optionnel : API FastAPI pour exposer le modèle
- Optionnel : suivi MLflow et Dockerfile pour déploiement

## Étapes principales

1. Nettoyage des données : traitement des NaN, outliers et incohérences
2. Préparation des variables : encodage, normalisation, sélection
3. Modélisation :
   - Régression linéaire
   - Random Forest
   - LightGBM
4. Évaluation : validation croisée, RMSE, MAE, R²
5. Analyse : comparaison des modèles et interprétation des résultats

## Bonnes pratiques

- Séparer train/test avant toute préparation finale
- Utiliser un pipeline reproductible
- Justifier les choix d'encodage et de normalisation
- Présenter clairement les métriques et leurs limites

## Notes

Ce README est un guide simple pour structurer le projet. Ajoutez les détails du dataset, les résultats et les étapes techniques dans le notebook et les fichiers annexes.

---

## Résumé de la pipeline

### 1. Analyse exploratoire (EDA)

- Aucune valeur manquante détectée
- 13 variables numériques, 1 cible (`valeur_mediane_logements`)
- Fortes corrélations entre variables démographiques (surface industrielle ↔ taux NOx, etc.)
- Nombreux outliers sur le taux de criminalité et l'indice de population — conservés car ils reflètent des réalités terrain
- Aucune donnée retirée

### 2. Prétraitement

- Imputation médiane sur les valeurs manquantes (pipeline générique)
- Normalisation `MinMaxScaler` sur toutes les variables numériques
- Pipeline `ColumnTransformer` conçu pour s'adapter automatiquement à l'arrivée de nouvelles variables catégorielles

### 3. Modèles testés

| Modèle | Type |
|---|---|
| Linear Regression | Linéaire |
| Ridge | Linéaire régularisé |
| Random Forest | Ensembliste |
| Gradient Boosting | Ensembliste |

### 4. Résultats — Validation simple (train/test 80/20)

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting** | 1.913 | 2.492 | 0.915 |
| Random Forest | 2.041 | 2.812 | 0.892 |
| Ridge | 3.112 | 4.859 | 0.678 |
| Linear Regression | 3.189 | 4.929 | 0.669 |

### 5. Résultats — Cross-Validation (5 folds)

| Modèle | RMSE moy | RMSE ± | R² moy |
|---|---|---|---|
| **Gradient Boosting** | 4.176 | ±1.105 | 0.675 |
| Random Forest | 4.418 | ±1.345 | 0.627 |
| Ridge | 5.465 | ±1.635 | 0.459 |
| Linear Regression | 5.829 | ±1.777 | 0.353 |

### 6. Conclusions

- **Gradient Boosting** est le meilleur modèle dans les deux évaluations
- La validation simple était **trop optimiste** (RMSE 2.492 vs 4.176 en CV) — signe d'overfitting sur les modèles ensemblistes
- Les modèles linéaires sont plus **stables** mais moins précis — ils généralisent mieux sur ce dataset
- Le dataset est **trop petit** (506 lignes) pour obtenir des folds CV vraiment représentatifs (± élevé sur tous les modèles)
- Erreur réelle estimée du meilleur modèle : **±4 176$** sur le prix d'un logement

## Dataset — Boston House Prices

Fichier : `data_set_boston_house.csv` — 14 variables, valeurs séparées par des virgules.

| Colonne | Type | Description |
|---------|------|-------------|
| `CRIM` | continu | Taux de criminalité par habitant dans la zone résidentielle |
| `ZN` | continu | Pourcentage de terrains résidentiels de plus de 25 000 pieds carrés |
| `INDUS` | continu | Pourcentage de surface occupée par des entreprises non commerciales (industrie légère) |
| `CHAS` | binaire (0/1) | Indicateur : la zone est-elle adjacente à la Charles River ? (1 = oui, 0 = non) |
| `NOX` | continu | Concentration en oxydes d'azote (parties par 10 millions) — indicateur de qualité de l'air |
| `RM` | continu | Nombre moyen de pièces par logement |
| `AGE` | continu | Proportion (%) de logements construits avant 1940 |
| `DIS` | continu | Distance pondérée aux cinq principaux centres d'emplois de Boston |
| `RAD` | entier | Indice d'accessibilité aux autoroutes radiales |
| `TAX` | continu | Taux d'imposition foncière par tranche de 10 000 dollars |
| `PTRATIO` | continu | Ratio moyen d'élèves par enseignant dans les écoles publiques |
| `B` | continu | Variable calculée : `1000(Bk - 0.63)²`, où Bk est la proportion de population noire |
| `LSTAT` | continu | Pourcentage de la population ayant un statut socio-économique faible |
| `MEDV` | continu | **Variable cible** — Valeur médiane des logements (en milliers de dollars) |
