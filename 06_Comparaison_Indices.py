# ============================================================
# 06_Comparaison_Indices.py
# COMPARAISON DES 4 METHODES DE CONSTRUCTION DE L'INDICE
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])
from pathlib import Path


# ============================================================
# 1. INDICES A COMPARER
# ============================================================
#
# NOTE (coordination avec la structure actuelle du memoire) :
# - "Index_Method1_Prix" / "Index_Method2_Log" n'existent plus :
#   01_Construction_Indice.py ne produit plus que Index_Base100
#   (voir la note dans export_workbook, "3 methodes de construction
#   alternatives... retirees"). construction_results n'a plus de
#   cle "comparison".
# - "RMT_HRP_Index" n'existe plus : le module RMT (desormais
#   04_RMT_Cleaning.py / run_rmt_cleaning()) ne construit plus
#   d'indice HRP, seulement Sigma_clean (entree de l'optimisation
#   convexe bi-penalisee).
# - 05_Convex_Optimization.py renvoie le portefeuille Min-Variance
#   bi-penalise sous la cle "index_erc_penalized" (serie nommee
#   "Index_ERC_Bipenalized") : c'est la cle qui est detectee ici
#   sous le nom "Index_ERC_Penalized". Les noms internes sont
#   conserves ; seul l'affichage (legende, statistiques) est renomme.
# ============================================================

TARGET_INDICES = [
    "Index_CAPI",
    "Index_Min_Variance",
    "Index_ERC",
    "Index_ERC_Penalized",
]

# Noms affichés (légende du graphique, tableau de statistiques)
DISPLAY_NAMES = {
    "Index_CAPI": "Pondération par capi",
    "Index_Min_Variance": "Min-Variance",
    "Index_ERC": "ERC",
    "Index_ERC_Penalized": "Min-Variance bi-pénalisé",
}


# ============================================================
# 2. RECHERCHE D'UNE SERIE
# ============================================================

def _find_index_series(result_sources, target_name):

    target_lower = target_name.lower()

    for source_name, result_dict in result_sources:

        if not isinstance(result_dict, dict):
            continue


        for key, value in result_dict.items():

            key_lower = str(key).lower()


            if isinstance(value, pd.Series):

                name_lower = (
                    str(value.name).lower()
                    if value.name is not None
                    else ""
                )


                if (
                    key_lower == target_lower
                    or name_lower == target_lower
                ):

                    print(
                        f"    → {target_name} trouvé dans "
                        f"{source_name}['{key}']"
                    )

                    return value



            elif isinstance(value, pd.DataFrame):

                if value.shape[1] == 1:

                    col_lower = str(
                        value.columns[0]
                    ).lower()


                    if (
                        key_lower == target_lower
                        or col_lower == target_lower
                    ):

                        print(
                            f"    → {target_name} trouvé dans "
                            f"{source_name}['{key}']"
                        )

                        return value.iloc[:,0]


    return None

# ============================================================
# 3. COLLECTE DES 4 INDICES
# ============================================================

def _collect_target_indices(result_sources):

    collected = {}
    missing = []

    print("\nRecherche des 4 indices...\n")

    for target_name in TARGET_INDICES:

        print(f"  Recherche : {target_name}")

        series = _find_index_series(
            result_sources,
            target_name
        )

        if series is None:

            print(
                f"    ✗ {target_name} NON TROUVÉ"
            )

            missing.append(target_name)
            continue

        series = pd.to_numeric(
            series,
            errors="coerce"
        )

        series = series.dropna()

        if len(series) < 20:

            print(
                f"    ✗ {target_name} : "
                f"pas assez d'observations ({len(series)})"
            )

            missing.append(
                f"{target_name} (pas assez d'observations)"
            )

            continue

        collected[target_name] = series

        print(
            f"    ✓ {target_name} "
            f"({len(series)} observations)"
        )

    return collected, missing


# ============================================================
# 4. ALIGNEMENT DES INDICES
# ============================================================

def _to_dataframe(indices_dict):

    if not indices_dict:
        return pd.DataFrame()

    df = pd.concat(
        indices_dict,
        axis=1
    )

    df.columns = list(
        indices_dict.keys()
    )

    # Suppression des colonnes dupliquées
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    return df


# ============================================================
# 5. REBASE A 100
# ============================================================

def _rebase_series_to_100(series):

    series = series.copy()

    valid = series.dropna()

    if len(valid) == 0:
        return series

    first_value = valid.iloc[0]

    if pd.isna(first_value) or first_value == 0:
        return series

    return (
        100
        *
        series
        /
        first_value
    )


# ============================================================
# 6. FONCTION PRINCIPALE
# ============================================================

def run_comparaison_indices(
    result_sources,
    chart_dir,
    rebase_100=True,
):

    chart_dir = Path(chart_dir)

    chart_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print("COMPARAISON DES 4 METHODES DE CONSTRUCTION")
    print("=" * 70)

    # ========================================================
    # COLLECTE
    # ========================================================

    indices_dict, missing = (
        _collect_target_indices(
            result_sources
        )
    )

    print("\n" + "=" * 70)
    print("RESULTAT DE LA DETECTION")
    print("=" * 70)

    print(
        f"\nIndices détectés : "
        f"{len(indices_dict)}/{len(TARGET_INDICES)}"
    )

    print("\nIndices disponibles :")

    for i, name in enumerate(
        indices_dict.keys(),
        start=1
    ):

        print(
            f"  {i}. ✓ {name}"
        )

    if missing:

        print(
            "\nIndices NON détectés :"
        )

        for name in missing:

            print(
                f"  ✗ {name}"
            )

    # ========================================================
    # VERIFICATION
    # ========================================================

    if not indices_dict:

        print(
            "\n[Comparaison] "
            "Aucun indice détecté."
        )

        return None

    # ========================================================
    # DATAFRAME
    # ========================================================

    df_all = _to_dataframe(
        indices_dict
    )

    # ========================================================
    # PERIODE GLOBALE
    # ========================================================

    print(
        f"\nPériode globale : "
        f"{df_all.index.min()} → "
        f"{df_all.index.max()}"
    )

    # ========================================================
    # PERIODE COMMUNE AUX 4 INDICES
    # ========================================================

    common_df = df_all.dropna(
        how="any"
    )

    if not common_df.empty:

        print(
            f"Période commune aux "
            f"{len(df_all.columns)} indices : "
            f"{common_df.index.min()} → "
            f"{common_df.index.max()}"
        )

    else:

        print(
            "\n⚠ Aucune période commune complète."
        )

    # ========================================================
    # REBASE A 100
    # ========================================================

    if rebase_100:

        df_plot = df_all.copy()

        for col in df_plot.columns:

            df_plot[col] = (
                _rebase_series_to_100(
                    df_plot[col]
                )
            )

    else:

        df_plot = df_all.copy()

    # ========================================================
    # GRAPHE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(18, 10)
    )

    # Palette qualitative dediee : jusqu'a 8 indices a comparer sur
    # ce graphique, PRIMARY_COLOR (couleur unique) les rendrait
    # indiscernables les uns des autres.
    n_series = len(df_plot.columns)

    if n_series <= 10:
        cmap = plt.cm.tab10
    elif n_series <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral

    colors = {
        col: cmap(i % cmap.N)
        for i, col in enumerate(df_plot.columns)
    }

    for col in df_plot.columns:

        ax.plot(
            df_plot.index,
            df_plot[col],
            label=DISPLAY_NAMES.get(col, col),
            linewidth=0.9,
            color=colors[col],
        )

    ax.axhline(
        100,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )

    ax.set_title(
        "Comparaison des méthodes "
        "de construction de l'indice bancaire panafricain",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Indice — Base 100"
    )

    ax.legend(
        loc="best",
        fontsize=8,
        ncol=2
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    output_path = (
        chart_dir
        /
        "Comparaison_Methodes_Construction.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # STATISTIQUES COMPARATIVES
    # ========================================================

    statistics = pd.DataFrame(
        index=df_plot.columns
    )

    statistics["N_Observations"] = (
        df_plot.notna().sum()
    )

    statistics["Niveau_Initial"] = (
        df_plot.apply(
            lambda x: x.dropna().iloc[0]
            if len(x.dropna()) > 0
            else float("nan")
        )
    )

    statistics["Niveau_Final"] = (
        df_plot.apply(
            lambda x: x.dropna().iloc[-1]
            if len(x.dropna()) > 0
            else float("nan")
        )
    )

    statistics["Performance_Totale_%"] = (
        (
            statistics["Niveau_Final"]
            /
            statistics["Niveau_Initial"]
            - 1
        )
        * 100
    )

    statistics["Volatilite"] = (
        df_plot.pct_change()
        .std()
    )

    statistics.index = [
        DISPLAY_NAMES.get(c, c) for c in statistics.index
    ]

    # ========================================================
    # AFFICHAGE FINAL
    # ========================================================

    print(
        f"\n✓ Graphe sauvegardé : "
        f"{output_path}"
    )

    print(
        f"\n✓ Nombre de courbes tracées : "
        f"{len(df_plot.columns)}"
    )

    print(
        "\nCourbes tracées :"
    )

    for i, col in enumerate(
        df_plot.columns,
        start=1
    ):

        print(
            f"  {i:2d}. {col}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "comparison_df": df_plot,

        "comparison_df_raw": df_all,

        "comparison_statistics": statistics,

        "chart_path": output_path,

        "detected_indices": list(
            df_plot.columns
        ),

        "missing_indices": missing,
    }