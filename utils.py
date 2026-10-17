"""
Infrastructure partagée du projet
Indice Bancaire Africain

Ce module centralise :
- les paramètres généraux ;
- le fichier Excel de sortie ;
- le dossier temporaire des graphiques ;
- le registre partagé des graphiques ;
- les fonctions de sauvegarde et de nettoyage.

Les modules :
    01_Construction_Indice.py
    02_FX_Adjusted_Index.py
    03_RMT_HRP_Index.py
    optimisation_convexe.py
    risque_systemique.py

peuvent utiliser cette infrastructure commune.
"""

import os
import shutil
import stat
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ============================================================
# 1. PARAMÈTRES GÉNÉRAUX
# ============================================================

# ------------------------------------------------------------
# Fichier Excel final
# ------------------------------------------------------------

OUTPUT_FILE = (
    "Analyse_Complete_Indice_Bancaire_Africain.xlsx"
)


# ------------------------------------------------------------
# Fichiers pour la pondération par capitalisation boursière
# (module 07_Calcul_Poids_CAPI.py)
# ------------------------------------------------------------
#
# CAPI_FILE :
#   Fichier Excel source contenant les capitalisations boursières
#   trimestrielles par banque. Format attendu : une colonne "Date"
#   (une ligne par trimestre) + une colonne par Ticker (même
#   convention de tickers que matriceJoursCours.xlsx / banks_info.xlsx).
#
# WEIGHTS_OUTPUT :
#   Fichier Excel dans lequel la matrice des poids CAPI calculés
#   (Date x Banque) est exportée par calculate_capi_weights().
#
# ATTENTION : adaptez CAPI_FILE au nom réel de votre fichier de
# capitalisations si celui-ci diffère.
# ------------------------------------------------------------

CAPI_FILE = "CAPI_FILE.xlsx"

BANKS_INFO_FILE = Path("banks_info.xlsx")

WEIGHTS_OUTPUT = "Poids_CAPI_Calcules.xlsx"

FX_FILE = Path("Taux.xlsx")

# ------------------------------------------------------------
# Dossier temporaire contenant les graphiques
# ------------------------------------------------------------

CHART_DIR = Path(
    "temp_charts_indice_africain"
)


# ------------------------------------------------------------
# Niveau de significativité
# ------------------------------------------------------------

ALPHA = 0.05


# ------------------------------------------------------------
# Nom utilisé pour l'Égypte
# ------------------------------------------------------------

EGYPT_NAME = "EGYPT"


# ============================================================
# 2. NETTOYAGE SÉCURISÉ DES GRAPHIQUES
# ============================================================

def remove_readonly(
    func,
    path,
    exc_info
):
    """
    Permet de supprimer un fichier ou dossier même
    si Windows le considère comme lecture seule.
    """

    try:

        os.chmod(
            path,
            stat.S_IWRITE
        )

        func(path)

    except Exception:

        pass


def safe_remove_directory(
    path
):
    """
    Supprime proprement un dossier temporaire.

    Parameters
    ----------
    path : str ou Path
        Dossier à supprimer.
    """

    path = Path(path)

    if not path.exists():

        return

    try:

        shutil.rmtree(
            path,
            onerror=remove_readonly
        )

    except Exception as e:

        print(
            "\nATTENTION : impossible de supprimer "
            f"le dossier temporaire {path}."
        )

        print(
            "Le programme continue quand même."
        )

        print(
            "Erreur :",
            e
        )


# ============================================================
# 3. INITIALISATION DU DOSSIER DES GRAPHIQUES
# ============================================================

"""
Le dossier des graphiques est créé s'il n'existe pas.

IMPORTANT :
On ne supprime PAS automatiquement le dossier ici.

Cela évite qu'un import de utils.py par un autre module
efface les graphiques déjà créés.

Le nettoyage complet est effectué explicitement
au début du programme principal à l'aide de :

    reset_chart_directory()
"""

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. REGISTRE PARTAGÉ DES GRAPHIQUES
# ============================================================

"""
Dictionnaire global contenant les graphiques générés
par les différents modules.

Exemple :

charts["01_Evolution_Indice"] =
    "temp_charts_indice_africain/01_Evolution_Indice.png"

charts["RMT_Spectrum"] =
    "temp_charts_indice_africain/RMT_Spectrum.png"

charts["HRP_Weights"] =
    "temp_charts_indice_africain/HRP_Weights.png"

Tous les modules utilisent donc le même registre.
"""

charts = {}


# ============================================================
# 5. SAUVEGARDE D'UN GRAPHIQUE
# ============================================================

def save_chart(
    name
):
    """
    Sauvegarde le graphique matplotlib courant.

    Parameters
    ----------
    name : str
        Nom du graphique sans extension.

    Returns
    -------
    str
        Chemin du fichier PNG créé.

    Notes
    -----
    Le graphique est :

    1. sauvegardé dans CHART_DIR ;
    2. ajouté au registre partagé charts ;
    3. fermé avec plt.close().
    """

    # --------------------------------------------------------
    # Sécurisation du nom
    # --------------------------------------------------------

    name = str(
        name
    ).strip()

    if not name:

        raise ValueError(
            "Le nom du graphique ne peut pas être vide."
        )


    # --------------------------------------------------------
    # Création du dossier si nécessaire
    # --------------------------------------------------------

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Chemin du fichier
    # --------------------------------------------------------

    path = (
        CHART_DIR
        /
        f"{name}.png"
    )


    # --------------------------------------------------------
    # Ajustement automatique
    # --------------------------------------------------------

    plt.tight_layout()


    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # Enregistrement dans le registre partagé
    # --------------------------------------------------------

    charts[name] = str(
        path
    )


    # --------------------------------------------------------
    # Fermeture de la figure
    # --------------------------------------------------------

    plt.close()


    return str(
        path
    )


# ============================================================
# 6. ENREGISTREMENT MANUEL D'UN GRAPHIQUE
# ============================================================

def register_chart(
    name,
    path
):
    """
    Enregistre manuellement un graphique dans le registre
    partagé.

    Utile lorsqu'un autre module a déjà créé et sauvegardé
    un fichier PNG.

    Parameters
    ----------
    name : str
        Nom du graphique.

    path : str ou Path
        Chemin vers le fichier PNG.
    """

    name = str(
        name
    ).strip()

    if not name:

        raise ValueError(
            "Le nom du graphique ne peut pas être vide."
        )


    path = Path(
        path
    )


    if not path.exists():

        print(
            f"ATTENTION : graphique introuvable : {path}"
        )

        return None


    charts[name] = str(
        path
    )


    return str(
        path
    )


# ============================================================
# 7. ENREGISTREMENT DE PLUSIEURS GRAPHIQUES
# ============================================================

def register_charts(
    chart_dictionary
):
    """
    Ajoute plusieurs graphiques au registre partagé.

    Parameters
    ----------
    chart_dictionary : dict
        Dictionnaire de la forme :

        {
            "Nom_1": "chemin/figure1.png",
            "Nom_2": "chemin/figure2.png"
        }
    """

    if chart_dictionary is None:

        return


    if not isinstance(
        chart_dictionary,
        dict
    ):

        raise TypeError(
            "chart_dictionary doit être un dictionnaire."
        )


    for name, path in (
        chart_dictionary.items()
    ):

        register_chart(
            name,
            path
        )


# ============================================================
# 8. RÉCUPÉRATION DU REGISTRE DES GRAPHIQUES
# ============================================================

def get_charts():
    """
    Retourne le registre partagé des graphiques.

    Returns
    -------
    dict
        Copie du dictionnaire charts.
    """

    return charts.copy()


# ============================================================
# 9. NETTOYAGE DU REGISTRE
# ============================================================

def clear_charts():
    """
    Vide uniquement le registre des graphiques.

    Les fichiers physiques ne sont pas supprimés.
    """

    charts.clear()


# ============================================================
# 10. RÉINITIALISATION COMPLÈTE
# ============================================================

def reset_chart_directory():
    """
    Supprime et recrée le dossier des graphiques.

    Le registre charts est également vidé.

    Cette fonction doit être appelée explicitement
    au début d'une nouvelle exécution complète du projet.
    """

    safe_remove_directory(
        CHART_DIR
    )


    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    charts.clear()


# ============================================================
# 11. VÉRIFICATION DE L'EXISTENCE D'UN GRAPHIQUE
# ============================================================

def chart_exists(
    name
):
    """
    Vérifie si un graphique est enregistré et existe
    physiquement.

    Parameters
    ----------
    name : str
        Nom du graphique.

    Returns
    -------
    bool
        True si le graphique existe, False sinon.
    """

    name = str(
        name
    ).strip()


    if name not in charts:

        return False


    return Path(
        charts[name]
    ).exists()


# ============================================================
# 12. INFORMATIONS SUR L'INFRASTRUCTURE
# ============================================================

def print_configuration():
    """
    Affiche la configuration actuelle du projet.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONFIGURATION DU PROJET"
    )

    print(
        "=" * 70
    )

    print(
        "OUTPUT_FILE :",
        OUTPUT_FILE
    )

    print(
        "CHART_DIR :",
        CHART_DIR
    )

    print(
        "ALPHA :",
        ALPHA
    )

    print(
        "EGYPT_NAME :",
        EGYPT_NAME
    )

    print(
        "Nombre de graphiques enregistrés :",
        len(charts)
    )

    print(
        "=" * 70
    )


# ============================================================
# 13. VÉRIFICATION DES GRAPHIQUES ENREGISTRÉS
# ============================================================

def print_charts():
    """
    Affiche la liste des graphiques actuellement enregistrés.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GRAPHIQUES ENREGISTRÉS"
    )

    print(
        "=" * 70
    )


    if not charts:

        print(
            "Aucun graphique enregistré."
        )

    else:

        for name, path in (
            charts.items()
        ):

            status = (
                "OK"
                if Path(path).exists()
                else "MANQUANT"
            )

            print(
                f"  [{status}] {name}"
            )

            print(
                f"       {path}"
            )


    print(
        "=" * 70
    )


# ============================================================
# 14. VÉRIFICATION DE L'INFRASTRUCTURE
# ============================================================

def check_infrastructure():
    """
    Vérifie que l'infrastructure de graphiques fonctionne
    correctement.

    Returns
    -------
    bool
        True si le dossier existe et est accessible.
    """

    try:

        CHART_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        test_path = (
            CHART_DIR
            /
            ".test_write"
        )


        with open(
            test_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "test"
            )


        test_path.unlink(
            missing_ok=True
        )


        return True


    except Exception as e:

        print(
            "\nERREUR : infrastructure des graphiques "
            "non fonctionnelle."
        )

        print(
            "Erreur :",
            e
        )

        return False


# ============================================================
# 15. TEST DU MODULE
# ============================================================

if __name__ == "__main__":

    print_configuration()

    print(
        "\nVérification de l'infrastructure..."
    )


    if check_infrastructure():

        print(
            "✓ Infrastructure fonctionnelle."
        )

    else:

        print(
            "✗ Problème détecté."
        )


def align_annual_weights_daily(poids_annuels, tickers, dates_index):
    """
    Étend les poids annuels de capi (une ligne par année) en une
    matrice quotidienne alignée sur le calendrier de l'indice.
    Chaque jour de bourse de l'année Y reçoit le poids calculé pour
    l'année Y : les poids sont donc rebalancés une fois par an.

    Parameters
    ----------
    poids_annuels : pd.DataFrame
        Index = année (int), colonnes = Ticker, valeurs = poids.
    tickers : list
        Liste des tickers dans l'ordre attendu (ex: returns_df.columns).
    dates_index : pd.DatetimeIndex
        Calendrier quotidien cible (ex: returns_df.index).

    Returns
    -------
    pd.DataFrame
        Index = dates_index, colonnes = tickers, valeurs = poids
        quotidiens rebalancés annuellement.
    """

    missing_tickers = [t for t in tickers if t not in poids_annuels.columns]

    if missing_tickers:
        raise ValueError(
            "Les tickers suivants sont absents des poids annuels de capi : "
            f"{missing_tickers}"
        )

    poids_annuels = poids_annuels[tickers].copy()
    poids_annuels = poids_annuels.sort_index()

    available_years = poids_annuels.index.tolist()
    min_year = min(available_years)
    max_year = max(available_years)

    daily_years = pd.Series(dates_index.year, index=dates_index)
    years_needed = sorted(daily_years.unique())

    out_of_range = [y for y in years_needed if y < min_year or y > max_year]

    if out_of_range:
        print(
            "ATTENTION : les années suivantes du calendrier de l'indice "
            f"sont hors de la plage disponible dans le fichier de capi "
            f"({min_year}-{max_year}) : {out_of_range}. "
            "Le poids de l'année disponible la plus proche sera utilisé."
        )

    poids_quotidien = pd.DataFrame(
        index=dates_index,
        columns=tickers,
        dtype=float
    )

    for year in years_needed:
        lookup_year = min(max(year, min_year), max_year)
        mask = (daily_years == year)
        poids_quotidien.loc[mask, :] = poids_annuels.loc[lookup_year].values

    row_sums = poids_quotidien.sum(axis=1)

    if not np.allclose(row_sums, 1):
        poids_quotidien = poids_quotidien.div(row_sums, axis=0)

    if poids_quotidien.isna().any().any():
        raise ValueError(
            "Des poids quotidiens sont manquants après alignement annuel."
        )

    return poids_quotidien

def compute_annual_capi_weights():
    """
    Calcule les poids annuels de capitalisation boursière (capi)
    pour chaque banque, à partir de CAPI_FILE.xlsx (capi en devise
    locale, une colonne par année) et de taux.xlsx (taux de change
    quotidiens, mêmes tickers que dans matriceJoursCours.xlsx).

    La conversion en devise commune utilise le dernier taux de change
    disponible au 31 décembre (ou avant) de chaque année.

    Returns
    -------
    dict
        {
            "capi_weights_quarterly": pd.DataFrame
                Index = année (int), colonnes = Ticker, valeurs = poids
                normalisés (somme = 1 par année).
            "capi_common_currency": pd.DataFrame
                Capitalisations converties en devise commune
                (avant normalisation), même structure.
        }
    """

    print("\nCalcul des poids annuels de capitalisation (capi)...")

    # ----------------------------------------------------------
    # 1. LECTURE DES FICHIERS
    # ----------------------------------------------------------

    # NOTE (coordination) : on utilise les constantes CAPI_FILE /
    # BANKS_INFO_FILE / FX_FILE definies en tete de utils.py plutot
    # que de re-coder les noms de fichiers en dur ici. Ca evite en
    # particulier un bug de casse : FX_FILE = Path("Taux.xlsx")
    # (T majuscule) alors que ce code lisait "taux.xlsx" (t
    # minuscule) -> FileNotFoundError garanti sous Linux, systeme de
    # fichiers sensible a la casse.
    capi_raw = pd.read_excel(CAPI_FILE)

    if "Ticker" not in capi_raw.columns:
        raise ValueError(f"La colonne 'Ticker' est absente de {CAPI_FILE}")

    info = pd.read_excel(BANKS_INFO_FILE)

    if "Devise" not in info.columns:
        raise ValueError(f"La colonne 'Devise' est absente de {BANKS_INFO_FILE}")

    fx = pd.read_excel(FX_FILE)

    if "Date" not in fx.columns:
        raise ValueError(f"La colonne 'Date' est absente de {FX_FILE}")

    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
    fx = fx.dropna(subset=["Date"])
    fx = fx.drop_duplicates(subset=["Date"], keep="last")
    fx.set_index("Date", inplace=True)
    fx = fx.apply(pd.to_numeric, errors="coerce")
    fx = fx.sort_index()

    # ----------------------------------------------------------
    # 2. IDENTIFICATION DES COLONNES ANNEE
    # ----------------------------------------------------------

    year_columns = [
        col for col in capi_raw.columns
        if col != "Ticker" and str(col).strip().isdigit()
    ]

    if not year_columns:
        raise ValueError(
            "Aucune colonne année détectée dans CAPI_FILE.xlsx "
            "(attendu : Ticker, 2019, 2020, ...)."
        )

    years = sorted(int(col) for col in year_columns)

    capi_raw = capi_raw.set_index("Ticker")
    capi_raw.columns = [
        int(col) if str(col).strip().isdigit() else col
        for col in capi_raw.columns
    ]

    tickers = capi_raw.index.tolist()

    missing_fx_tickers = [t for t in tickers if t not in fx.columns]
    if missing_fx_tickers:
        raise ValueError(
            "Taux de change absents pour les banques suivantes "
            f"(capi) : {missing_fx_tickers}"
        )

    # ----------------------------------------------------------
    # 3. TAUX DE CHANGE DE FIN D'ANNEE
    # ----------------------------------------------------------

    year_end_fx = {}

    for year in years:
        cutoff = pd.Timestamp(year=year, month=12, day=31)
        available_dates = fx.index[fx.index <= cutoff]

        if len(available_dates) == 0:
            raise ValueError(
                f"Aucun taux de change disponible avant ou au "
                f"31/12/{year} dans taux.xlsx."
            )

        last_date = available_dates.max()
        year_end_fx[year] = fx.loc[last_date]

    # ----------------------------------------------------------
    # 4. CONVERSION EN DEVISE COMMUNE
    # ----------------------------------------------------------

    capi_common = pd.DataFrame(index=tickers, columns=years, dtype=float)

    for year in years:
        rates = year_end_fx[year].reindex(tickers)

        if rates.isna().any():
            missing = rates[rates.isna()].index.tolist()
            raise ValueError(
                f"Taux de change manquant pour l'année {year} "
                f"pour les banques : {missing}"
            )

        capi_common[year] = capi_raw[year] * rates

    # ----------------------------------------------------------
    # 5. VERIFICATION
    # ----------------------------------------------------------

    if capi_common.isna().any().any():
        missing = capi_common.columns[capi_common.isna().any()].tolist()
        raise ValueError(
            f"Des capitalisations sont manquantes pour les années : {missing}"
        )

    if (capi_common <= 0).any().any():
        raise ValueError("Certaines capitalisations sont nulles ou négatives.")

    # ----------------------------------------------------------
    # 6. NORMALISATION ANNUELLE
    # ----------------------------------------------------------

    capi_weights = capi_common.div(capi_common.sum(axis=0), axis=1)

    capi_weights_quarterly = capi_weights.T
    capi_weights_quarterly.index.name = "Year"

    capi_common_currency = capi_common.T
    capi_common_currency.index.name = "Year"

    sums = capi_weights_quarterly.sum(axis=1)

    if not np.allclose(sums, 1):
        raise ValueError(
            "La somme des poids annuels de capi n'est pas égale à 1 "
            f"pour certaines années : {sums[~np.isclose(sums, 1)].to_dict()}"
        )

    print(
        f"✓ Poids de capi calculés pour {len(years)} années "
        f"({years[0]}–{years[-1]}) et {len(tickers)} banques."
    )

    return {
        "capi_weights_quarterly": capi_weights_quarterly,
        "capi_common_currency": capi_common_currency,
    }