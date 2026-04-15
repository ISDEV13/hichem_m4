import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

def load_data(path):
    df = pd.read_csv(path)
    return df



def analyse_missing_values(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    msno.matrix(df, ax=axes[0])
    axes[0].set_title("Matrice des valeurs manquantes")

    msno.bar(df, ax=axes[1])
    axes[1].set_title("Nombre de valeurs manquantes")

    msno.heatmap(df, ax=axes[2])
    axes[2].set_title("Corrélation des valeurs manquantes")

    plt.tight_layout()
    plt.show()


def analyse_distributions(df):
    colonnes = df.select_dtypes(include='number').columns

    fig, axes = plt.subplots(len(colonnes), 1, figsize=(8, 4 * len(colonnes)))

    if len(colonnes) == 1:
        axes = [axes]

    for ax, col in zip(axes, colonnes):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")

    plt.tight_layout()
    plt.show()


def analyse_correlations(df):
    
  # Séparer types
    df_num = df.select_dtypes(include='number')
  
    df_corr = df_num

    # Calcul corrélation
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation (numériques)")
    plt.show()


def analyse_boxplots(df):
      # Sélection des colonnes numériques
    df_num = df.select_dtypes(include='number')

    # Boxplot
    plt.figure(figsize=(12, 6))
    df_num.boxplot()

    plt.title("Boxplots des variables numériques")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
def analyse_categorical_distributions(df, to_drop=None):
    df = df.copy()
    if to_drop is not None:
            df = df.drop(columns=to_drop, errors="ignore")
    # Sélection des colonnes catégorielles
    df_cat = df.select_dtypes(include=['object', 'category', 'bool'])

    n_cols = 3  # nombre de graphes par ligne
    n_rows = (len(df_cat.columns) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i, col in enumerate(df_cat.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution de {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    