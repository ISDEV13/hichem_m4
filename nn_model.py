# =============================================================================
#  MODÈLE DE RÉSEAU DE NEURONES
#  Création, entraînement et inférence du modèle Keras.
# =============================================================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ── Création ──────────────────────────────────────────────────────────────────

def create_nn_model(input_dim):
    """
    Construit et compile un réseau de neurones dense à 2 couches cachées.

    Architecture :
        Dense(64, relu) → Dense(32, relu) → Dense(1)
    Optimiseur : Adam   |   Loss : MSE
    """
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_model(
    model,
    X, y,
    X_val=None, y_val=None,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=None,
):
    """
    Entraîne le modèle et retourne (model, history).

    Si X_val et y_val sont fournis, la validation est activée à chaque époque.
    """
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    history = model.fit(
        X, y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    return model, history


# ── Prédiction ────────────────────────────────────────────────────────────────

def model_predict(model, X):
    """Retourne les prédictions aplaties en tableau 1-D."""
    return model.predict(X).flatten()


# ── Comparaison de modèles ────────────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test, epochs=50):
    """
    Entraîne et compare plusieurs modèles de régression sur les mêmes données.

    Modèles testés :
        Linear Regression, Ridge, Random Forest, Gradient Boosting, Neural Network

    Retourne un DataFrame trié par RMSE croissant.
    """
    modeles = {
        # Modèle de base : trace une droite entre les features et la cible.
        # Rapide et interprétable, mais suppose une relation linéaire.
        # Sensible aux outliers et au multicolinéarité.
        "Linear Regression":    LinearRegression(),

        # Variante de la régression linéaire avec pénalité L2 sur les poids.
        # Réduit l'overfitting quand les features sont corrélées entre elles.
        # Préférable à LinearRegression quand le dataset a beaucoup de features.
        "Ridge":                Ridge(),

        # Ensemble de centaines d'arbres de décision entraînés en parallèle.
        # Très robuste aux outliers et aux relations non-linéaires.
        # Pas besoin de normaliser les données. Peut overfitter sur petits datasets.
        "Random Forest":        RandomForestRegressor(random_state=42),

        # Arbres construits séquentiellement : chaque arbre corrige les erreurs du précédent.
        # Souvent le meilleur modèle sur données tabulaires.
        # Plus lent à entraîner que Random Forest, mais généralement plus précis.
        "Gradient Boosting":    GradientBoostingRegressor(random_state=42),
    }

    resultats = {}

    # Modèles sklearn
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)
        resultats[nom] = {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²":   r2_score(y_test, y_pred),
        }



    df_resultats = pd.DataFrame(resultats).T.sort_values("RMSE")

    output_dir = "analyse_model"
    os.makedirs(output_dir, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(output_dir, f"run_{run_date}.log")
    # On capture l'ID du handler pour pouvoir le supprimer après écriture
    handler_id = logger.add(log_path, format="{time} | {level} | {message}", level="INFO")

    # INFO = validation simple train/test (résultat indicatif)
    logger.info(f"[VALIDATION SIMPLE] Run du {run_date}")
    logger.info(f"{'Modèle':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    logger.info("-" * 55)
    for nom, row in df_resultats.iterrows():
        logger.info(f"{nom:<25} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} {row['R²']:>8.3f}")
    # WARNING pour faire ressortir la conclusion dans le log
    logger.warning(f"Meilleur modèle : {df_resultats.index[0]} (RMSE = {df_resultats['RMSE'].iloc[0]:.3f})")

    # Supprime le handler → le prochain run n'écrira plus dans ce fichier
    logger.remove(handler_id)

    return df_resultats


# ── Cross-Validation ──────────────────────────────────────────────────────────

def compare_cv(X, y, cv=5):
    """
    Compare plusieurs modèles de régression via cross-validation.

    Paramètres :
        X  : features complètes (pas splittées)
        y  : cible
        cv : nombre de folds (défaut 5)

    Retourne un DataFrame trié par RMSE moyen croissant.
    """
    modeles = {
        # Modèle de base : trace une droite entre les features et la cible.
        # Rapide et interprétable, mais suppose une relation linéaire.
        # Sensible aux outliers et à la multicolinéarité.
        "Linear Regression":  LinearRegression(),

        # Variante de la régression linéaire avec pénalité L2 sur les poids.
        # Réduit l'overfitting quand les features sont corrélées entre elles.
        # Préférable à LinearRegression quand le dataset a beaucoup de features.
        "Ridge":              Ridge(),

        # Ensemble de centaines d'arbres de décision entraînés en parallèle.
        # Très robuste aux outliers et aux relations non-linéaires.
        # Pas besoin de normaliser les données. Peut overfitter sur petits datasets.
        "Random Forest":      RandomForestRegressor(random_state=42),

        # Arbres construits séquentiellement : chaque arbre corrige les erreurs du précédent.
        # Souvent le meilleur modèle sur données tabulaires.
        # Plus lent à entraîner que Random Forest, mais généralement plus précis.
        "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
    }

    # Dictionnaire vide qui va accumuler les résultats de chaque modèle
    resultats = {}

    for nom, modele in modeles.items():
        # cross_val_score découpe X et y en cv=5 morceaux (folds),
        # entraîne le modèle sur 4 folds et teste sur le 5ème,
        # répète 5 fois en changeant le fold de test à chaque tour.
        # scoring="neg_root_mean_squared_error" → sklearn retourne des valeurs
        # négatives par convention, le - devant remet en positif.
        scores_rmse = -cross_val_score(modele, X, y, cv=cv, scoring="neg_root_mean_squared_error")

        # Même principe pour le MAE
        scores_mae  = -cross_val_score(modele, X, y, cv=cv, scoring="neg_mean_absolute_error")

        # R² n'a pas besoin du - car il est naturellement positif (entre 0 et 1)
        scores_r2   =  cross_val_score(modele, X, y, cv=cv, scoring="r2")

        resultats[nom] = {
            # Moyenne des 5 RMSE → score global du modèle
            "RMSE moy":  scores_rmse.mean(),
            # Écart-type des 5 RMSE → mesure la stabilité du modèle
            # Un ± élevé = le modèle est instable selon les données
            "RMSE ±":    scores_rmse.std(),
            "MAE moy":   scores_mae.mean(),
            "R² moy":    scores_r2.mean(),
        }

    # Transforme le dictionnaire en DataFrame puis trie par RMSE moyen
    # .T = transpose → les modèles deviennent les lignes, les métriques les colonnes
    df_resultats = pd.DataFrame(resultats).T.sort_values("RMSE moy")

    # Crée le dossier de logs s'il n'existe pas encore
    output_dir = "analyse_model"
    os.makedirs(output_dir, exist_ok=True)

    # Horodatage du run pour nommer le fichier de log de façon unique
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Préfixe "cv_" pour distinguer des logs de compare_models
    log_path = os.path.join(output_dir, f"cv_{run_date}.log")
    # On capture l'ID du handler pour pouvoir le supprimer après écriture
    handler_id = logger.add(log_path, format="{time} | {level} | {message}", level="INFO")

    # SUCCESS = cross-validation (résultat plus rigoureux que la validation simple)
    logger.success(f"[CROSS-VALIDATION {cv} FOLDS] Run du {run_date}")
    logger.success(f"{'Modèle':<25} {'RMSE moy':>10} {'RMSE ±':>8} {'MAE moy':>10} {'R² moy':>8}")
    logger.success("-" * 67)
    for nom, row in df_resultats.iterrows():
        logger.success(f"{nom:<25} {row['RMSE moy']:>10.3f} {row['RMSE ±']:>8.3f} {row['MAE moy']:>10.3f} {row['R² moy']:>8.3f}")
    # index[0] = première ligne = meilleur modèle car trié par RMSE croissant
    # WARNING pour faire ressortir la conclusion dans le log
    logger.warning(f"Meilleur modèle : {df_resultats.index[0]} (RMSE moy = {df_resultats['RMSE moy'].iloc[0]:.3f} ± {df_resultats['RMSE ±'].iloc[0]:.3f})")

    # Supprime le handler → le prochain run n'écrira plus dans ce fichier
    logger.remove(handler_id)

    return df_resultats
