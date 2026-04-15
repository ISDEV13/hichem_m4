from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd


def preprocessingTechnique(df, target_col, to_drop=None):
    df = df.copy()
    rapport = []

    # Suppression des colonnes inutiles
    if to_drop:
        dropped = [c for c in to_drop if c in df.columns]
        df = df.drop(columns=to_drop, errors="ignore")
        rapport.append(f"Colonnes supprimées ({len(dropped)}) : {dropped}")

    # Détection automatique des colonnes (hors cible)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    rapport.append(f"Colonnes numériques ({len(num_cols)}) : {num_cols}")
    rapport.append(f"Colonnes catégorielles ({len(cat_cols)}) : {cat_cols}")

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Assemblage (uniquement les types présents)
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
        rapport.append(f"Numériques → imputation médiane + MinMaxScaler")
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
        rapport.append(f"Catégorielles → imputation mode + OneHotEncoder")

    preprocessor = ColumnTransformer(transformers)

    X_processed = preprocessor.fit_transform(X)
    
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    df_final = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    # Vérification valeurs manquantes
    nan_avant = df.drop(columns=[target_col]).isnull().sum().sum()
    nan_apres = X_processed.isnull().sum().sum()
    rapport.append(f"Valeurs manquantes avant : {nan_avant} → après : {nan_apres}")

    # Dimensions
    rapport.append(f"Dimensions avant  : {df.drop(columns=[target_col]).shape}")
    rapport.append(f"Dimensions après  : {X_processed.shape}")

    # Affichage
    print("=" * 55)
    print("        RAPPORT DE PREPROCESSING")
    print("=" * 55)
    for ligne in rapport:
        print(f"  • {ligne}")
    print("=" * 55)

    return X_processed, y, preprocessor, df_final


def split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        test_size    : proportion du jeu de test (défaut 20 %)
        random_state : graine pour la reproductibilité

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
