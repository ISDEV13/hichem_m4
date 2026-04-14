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
