# -*- coding: utf-8 -*-
"""
07_Calcul_Poids_CAPI.py

Module de calcul des poids des banques selon la méthode de
pondération par capitalisation boursière (CAPI), avec
REBALANCEMENT QUOTIDIEN, et construction de l'indice pondéré par
ces poids (Méthode CAPI, Base 100).

Principe (version rebalancement quotidien)
-------------------------------------------
La capitalisation boursière annuelle (donnée d'entrée, une valeur
par banque et par année) sert uniquement de point d'ancrage. Entre
deux ancrages annuels, la capitalisation de chaque banque est
"diffusée" jour après jour à partir des rendements log quotidiens
déjà nettoyés (returns_df) :

    CAPI(i, t) = CAPI(i, ancre) * exp( somme_{ancre < k <= t} r(i,k) )

Le poids est ensuite recalculé CHAQUE JOUR (et non plus reconduit
tel quel pendant un an) :

    w(i, t) = CAPI(i, t) / somme_j( CAPI(j, t) )

C'est la différence essentielle avec l'ancienne version du module,
qui calculait un poids une seule fois par année et le reconduisait
à l'identique (forward-fill) jusqu'à l'année suivante : ici, le
poids de chaque banque évolue quotidiennement avec sa performance
relative, tout en repartant de la capitalisation réelle observée à
chaque nouvelle ancre annuelle.

Ce module :
    1. Lit le fichier Excel des capitalisations boursières annuelles
       (chemin défini dans utils.py -> CAPI_FILE)
    2. Nettoie et contrôle les données (NaN, valeurs négatives, doublons,
       tri chronologique, banques manquantes sur certaines années)
    3. Convertit les capitalisations en devise commune (fichier FX)
    4. Calcule les poids ANNUELS (calculate_capi_weights()), exportés tels
       quels pour référence / reporting (composition de fin d'année)
    5. Diffuse quotidiennement la capitalisation à partir de ces ancres
       annuelles en utilisant les rendements quotidiens, et RECALCULE le
       poids CHAQUE JOUR (build_capi_weighted_index()) pour construire
       l'indice pondéré par capitalisation, avec ses propres graphiques
       (via save_chart, registre partagé avec les autres modules).

CONVENTION FX (coordination avec le reste du projet) :
-------------------------------------------------------
Comme dans 01_Construction_Indice.py (`prices_common = prices * fx`)
et dans utils.compute_annual_capi_weights(), le fichier de taux de
change (utils.FX_FILE, "taux.xlsx") est indexé PAR TICKER (une
colonne par banque, pas par code devise) et se MULTIPLIE aux
valeurs locales pour obtenir la devise commune :

    CAPI_commune(i, t) = CAPI_locale(i, t) * FX(i, t)

Il n'y a donc pas besoin d'un mapping ticker -> devise (banks_info /
colonne "Devise") : chaque banque a directement sa propre colonne
dans taux.xlsx, exactement comme pour les cours (matriceJoursCours.xlsx).

Ce module ne touche ni au classeur Excel ni au Dashboard :
01_Construction_Indice.py (export_workbook) pilote l'export complet
et intègre ces résultats, comme pour les autres méthodes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])

# Import des constantes du projet (chemins, paramètres globaux)
from utils import (
    CAPI_FILE,
    FX_FILE,
    WEIGHTS_OUTPUT,
    save_chart,
)

# Tolérance numérique utilisée pour vérifier que la somme des poids
# par date est bien égale à 1 (arrondis flottants)
TOLERANCE_SOMME_POIDS = 1e-6


def _lire_donnees_capi(chemin_fichier: Path) -> pd.DataFrame:
    """
    Lecture du fichier CAPI annuel.

    Format attendu :

    Ticker | 2016 | 2017 | ... | 2026

    Retour :
    DataFrame Banque x Année
    """

    chemin_fichier = Path(chemin_fichier)

    if not chemin_fichier.exists():
        raise FileNotFoundError(
            f"Fichier CAPI introuvable : {chemin_fichier}"
        )

    df = pd.read_excel(
        chemin_fichier,
        index_col=0
    )

    # Conversion des années en colonnes numériques
    df.columns = df.columns.astype(int)

    return df


def _lire_taux_change(chemin_fichier: Path) -> pd.DataFrame:
    """
    Lecture du fichier FX quotidien.

    Format attendu :

    Date | MAD | ZAR | EGP | ...
    """

    fx = pd.read_excel(
        chemin_fichier
    )

    fx["Date"] = pd.to_datetime(
        fx["Date"]
    )

    fx = fx.set_index("Date")

    fx = fx.sort_index()

    return fx


def _convertir_fx_annuel(fx):
    """
    Transforme les taux quotidiens en taux annuels.
    On prend le dernier taux disponible de chaque année.
    """

    fx_annuel = (
        fx
        .resample("YE")
        .last()
    )

    fx_annuel.index = fx_annuel.index.year

    return fx_annuel


def _nettoyer_donnees_capi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les règles de nettoyage définies pour le module :
        - suppression des doublons de dates (on garde la dernière occurrence)
        - tri chronologique des dates
        - remplacement des NaN par 0
        - remplacement des capitalisations négatives par 0
          (une capitalisation boursière ne peut pas être négative :
          on considère ces valeurs comme des erreurs de saisie)
    """
    # Suppression des doublons éventuels sur l'index Date
    # (on conserve la dernière ligne rencontrée pour une date donnée)
    if df.index.duplicated().any():
        print("Attention : doublons de dates détectés, suppression des doublons (dernière valeur conservée).")
        df = df[~df.index.duplicated(keep="last")]

    # Tri chronologique des dates (les années doivent être ordonnées)
    df = df.sort_index()

    # Remplacement des valeurs manquantes par 0
    # (une banque sans donnée de capitalisation sur une année donnée
    # est considérée comme absente de l'indice sur cette année)
    df = df.fillna(0)

    # Sécurité : aucune capitalisation ne doit être négative.
    # On ramène les valeurs négatives à 0 pour ne jamais produire
    # de poids négatifs par la suite.
    if (df < 0).any().any():
        print("Attention : capitalisations négatives détectées, remplacées par 0.")
        df = df.clip(lower=0)

    return df


def _convertir_capi_devise_commune(
        df_capi: pd.DataFrame,
        fx_annuel: pd.DataFrame
) -> pd.DataFrame:
    """
    Conversion des capitalisations locales en devise commune,
    banque par banque, en utilisant la colonne de taux de change
    propre à CHAQUE TICKER (fx_annuel est indexé par ticker, pas par
    code devise — même convention que taux.xlsx dans tout le reste
    du projet) :

        CAPI_commune(i, t) = CAPI_locale(i, t) * FX(i, t)

    C'est la même convention que `prices_common = prices * fx` dans
    01_Construction_Indice.py et que utils.compute_annual_capi_weights()
    (capi_common[year] = capi_raw[year] * rates) : on MULTIPLIE, on ne
    divise pas, et on n'a pas besoin d'un mapping ticker -> devise.

    NB (coordination) : l'ancien fichier définissait deux fois une
    fonction `_convertir_capi_usd` (la seconde écrasait la première
    silencieusement) et supposait, à tort, un fichier FX indexé par
    code devise (MAD, ZAR, EGP...) sur lequel on divisait. Ce n'est
    pas le format réel de utils.FX_FILE (taux.xlsx) tel qu'utilisé
    par le reste du pipeline : c'est corrigé ici.
    """

    df_commune = df_capi.copy()

    for banque in df_capi.index:

        if banque not in fx_annuel.columns:
            raise ValueError(
                f"Taux de change absent pour {banque} "
                f"(colonne attendue dans {FX_FILE})"
            )

        for annee in df_capi.columns:

            if annee not in fx_annuel.index:
                raise ValueError(
                    f"Année {annee} absente du fichier FX"
                )

            taux = fx_annuel.loc[annee, banque]

            df_commune.loc[banque, annee] = (
                df_capi.loc[banque, annee] * taux
            )

    return df_commune


def _calculer_poids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les poids CAPI par date :
        w(i,t) = CAPI(i,t) / somme_j( CAPI(j,t) )

    Si la somme des capitalisations d'une date est nulle (toutes les
    banques absentes ou à 0 sur cette date), les poids sont fixés à 0
    pour cette date afin d'éviter une division par zéro.
    """
    somme_par_date = df.sum(axis=1)

    # Division ligne par ligne ; on évite la division par zéro en
    # remplaçant temporairement les sommes nulles par NaN (le résultat
    # sera de toute façon 0 puisque le numérateur vaut 0 dans ce cas)
    somme_sans_zero = somme_par_date.replace(0, np.nan)

    poids = df.div(somme_sans_zero, axis=0)

    # Les dates où la somme était nulle donnent des NaN après division :
    # on les remplace par 0 (aucun poids ne peut être attribué)
    poids = poids.fillna(0)

    # Sécurité finale : aucun poids négatif ne doit subsister
    poids = poids.clip(lower=0)

    return poids


def _verifier_somme_poids(poids: pd.DataFrame) -> None:
    """
    Vérifie que la somme des poids par date vaut 1 (à tolérance près).
    Les dates dont la capitalisation totale est nulle sont ignorées
    (somme des poids = 0 dans ce cas, ce qui est attendu).
    """
    sommes = poids.sum(axis=1)

    for date, somme in sommes.items():
        if somme == 0:
            # Date sans aucune capitalisation disponible : normal
            continue
        if abs(somme - 1.0) > TOLERANCE_SOMME_POIDS:
            print(
                f"Attention : la somme des poids pour la date {date} "
                f"vaut {somme:.6f} au lieu de 1."
            )


def _exporter_poids(poids: pd.DataFrame, chemin_sortie: Path) -> None:
    """
    Exporte la matrice des poids (Date x Banque) dans un fichier Excel.
    Crée le dossier de destination si nécessaire.
    """
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    poids.to_excel(chemin_sortie, index=True, sheet_name="Poids_CAPI")


# ================================================================
# CAPITALISATION ANNUELLE EN USD (ANCRES DE REBALANCEMENT)
# ================================================================

def calculate_capi_annual_usd() -> pd.DataFrame:
    """
    Lit, convertit (en devise commune) et nettoie la capitalisation
    boursière annuelle par banque, et renvoie une matrice
    Année x Banque dont l'index a été décalé au 1er janvier de
    l'année suivante (ancre de rebalancement) :

        capi(i, "01/01/N+1") = capitalisation de i au 31/12/N

    Cette matrice sert de point de départ ("ancre") à la diffusion
    quotidienne de la capitalisation (cf. _construire_capi_quotidien_drift).

    Le nom de la fonction (suffixe "_usd") est conservé pour la
    compatibilité de l'API avec le reste du projet ; la devise
    commune réellement obtenue est celle du fichier utils.FX_FILE
    (taux.xlsx), la même que celle utilisée pour prices_common dans
    01_Construction_Indice.py.
    """

    print("Lecture des données CAPI...")

    df_capi = _lire_donnees_capi(CAPI_FILE)

    print("Lecture des taux FX...")

    fx = _lire_taux_change(FX_FILE)
    fx_annuel = _convertir_fx_annuel(fx)

    print("Conversion CAPI locale -> devise commune...")

    df_capi = _convertir_capi_devise_commune(df_capi, fx_annuel)

    print("\nDEBUG CAPI APRES FX")
    print(
        df_capi.loc[
            ["ATW MC Equity",
            "SBK SJ Equity",
            "CPI SJ Equity"],
            [2024,2025,2026]
        ]
    )

    # Banque x Année  --->  Année x Banque
    df_capi = df_capi.T

    df_capi = _nettoyer_donnees_capi(df_capi)

    # ------------------------------------------------------------
    # Conversion des années (int) en dates d'ancrage : la
    # capitalisation observée pour l'année Y s'applique à partir du
    # 1er janvier de l'année Y+1 (31/12/Y + 1 jour).
    # ------------------------------------------------------------

    df_capi.index = (
        pd.to_datetime(df_capi.index.astype(str) + "-12-31")
        + pd.Timedelta(days=1)
    )

    return df_capi


def calculate_capi_weights() -> pd.DataFrame:
    """
    Calcule les poids CAPI ANNUELS (une ligne par ancre du 1er
    janvier) à partir de la capitalisation en USD, les exporte
    (WEIGHTS_OUTPUT) et les renvoie.

    Ces poids annuels servent de référence / reporting (composition
    de fin d'année, HHI, Top 5/10...). Ils ne sont PLUS utilisés tels
    quels pour construire l'indice quotidien : voir
    build_capi_weighted_index(), qui applique désormais un
    rebalancement quotidien.
    """

    df_capi = calculate_capi_annual_usd()
    print("\n========== CONTROLE CAPITALISATIONS APRES FX ==========")

    print(
        df_capi.loc[
            [
                "ATW MC Equity",
                "SBK SJ Equity",
                "GTCO NL Equity",
                "FIRSTHLD NL Equity"
            ]
        ]
    )

    print("\nRatios ATW / grandes banques :")

    print(
        df_capi.loc["ATW MC Equity"] /
        df_capi.loc["SBK SJ Equity"]
    )

    print("========================================================")

    print("\nCAPITALISATION USD :")
    print(
        df_capi[
            [
                "ATW MC Equity",
                "SBK SJ Equity",
                "GTCO NL Equity",
                "FIRSTHLD NL Equity"
            ]
        ].tail()
)
    print("Calcul des poids CAPI (annuels, référence)...")

    df_poids = _calculer_poids(df_capi)

    _verifier_somme_poids(df_poids)

    _exporter_poids(df_poids, WEIGHTS_OUTPUT)

    print("Export des poids terminé.")

    return df_poids


# ================================================================
# DIFFUSION QUOTIDIENNE DE LA CAPITALISATION (REBALANCEMENT QUOTIDIEN)
# ================================================================

def _construire_capi_quotidien_drift(capi_annuels_usd: pd.DataFrame,
                                      returns_df: pd.DataFrame,
                                      tickers_cibles: list) -> pd.DataFrame:
    """
    Construit la capitalisation boursière QUOTIDIENNE (USD) par
    diffusion des rendements log quotidiens à partir de la dernière
    ancre annuelle connue :

        CAPI(i, t) = CAPI(i, ancre) * exp( somme_{ancre < k <= t} r(i, k) )

    où "ancre" est la dernière date du 1er janvier (issue de
    capi_annuels_usd.index) inférieure ou égale à t.

    Contrairement à un simple forward-fill du poids annuel, la
    capitalisation ainsi obtenue évolue CHAQUE JOUR avec la
    performance propre de chaque banque : le poids qui en découle
    (cf. _poids_quotidien_depuis_capi) est donc recalculé
    quotidiennement, tout en repartant de la capitalisation réellement
    observée à chaque nouvelle ancre annuelle.

    - Tickers présents dans returns_df mais absents de capi_annuels_usd :
      écartés (avertissement), impossible de leur définir une ancre.
    - Tickers présents dans capi_annuels_usd mais absents de returns_df :
      simplement ignorés (non utilisables dans l'indice quotidien).
    - Dates antérieures à la première ancre disponible : capitalisation
      non définissable -> NaN (exclues ensuite de la construction de
      l'indice, comme dans l'ancienne version).

    Returns
    -------
    pandas.DataFrame
        Capitalisation quotidienne (Date x Ticker) en USD, alignée sur
        l'index de returns_df, limitée aux tickers communs.
    """

    tickers_communs = [
        t for t in tickers_cibles
        if t in capi_annuels_usd.columns
    ]

    tickers_manquants = [
        t for t in tickers_cibles
        if t not in capi_annuels_usd.columns
    ]

    if tickers_manquants:
        print(
            "Attention : banques absentes du fichier CAPI "
            f"(capitalisation fixée à 0) : {tickers_manquants}"
        )

    if not tickers_communs:
        raise ValueError(
            "Aucun ticker commun entre le fichier CAPI et les "
            "rendements de l'indice : impossible de construire "
            "l'indice pondéré par capitalisation."
        )

    ancres = capi_annuels_usd.index.sort_values()
    index_quotidien = returns_df.index

    capi_quotidien = pd.DataFrame(
        index=index_quotidien,
        columns=tickers_communs,
        dtype=float
    )

    # Rendements manquants ponctuels traités comme "pas de variation"
    # (0), pour ne pas interrompre la diffusion d'une banque sur un
    # jour isolé sans cotation.
    rendements_communs = returns_df[tickers_communs].fillna(0.0)

    for i, ancre in enumerate(ancres):

        borne_sup = ancres[i + 1] if i + 1 < len(ancres) else None

        if borne_sup is not None:
            masque = (index_quotidien >= ancre) & (index_quotidien < borne_sup)
        else:
            masque = index_quotidien >= ancre

        if not masque.any():
            continue

        dates_periode = index_quotidien[masque]

        cap_ancre = capi_annuels_usd.loc[ancre, tickers_communs]

        # Rendement cumulé depuis le début de la période (l'ancre),
        # recalculé chaque jour -> diffusion quotidienne de la capi.
        rendements_cumules = (
            rendements_communs
            .loc[dates_periode]
            .cumsum()
        )

        capi_quotidien.loc[dates_periode] = (
            cap_ancre.values * np.exp(rendements_cumules.values)
        )

    return capi_quotidien.astype(float)


def _poids_quotidien_depuis_capi(capi_quotidien: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule le poids CAPI CHAQUE JOUR à partir de la capitalisation
    quotidienne diffusée :

        w(i, t) = CAPI(i, t) / somme_j( CAPI(j, t) )
    """

    somme_quotidienne = capi_quotidien.sum(axis=1)
    somme_sans_zero = somme_quotidienne.replace(0, np.nan)

    poids_quotidien = capi_quotidien.div(somme_sans_zero, axis=0)

    return poids_quotidien


# ================================================================
# CONSTRUCTION DE L'INDICE PONDERE PAR CAPITALISATION (BASE 100)
# ================================================================

def build_capi_weighted_index(returns_df: pd.DataFrame,
                               capi_annuels_usd: pd.DataFrame):
    """
    Construit l'indice pondéré par capitalisation boursière, avec
    REBALANCEMENT QUOTIDIEN des poids, à partir des rendements log
    quotidiens déjà nettoyés (returns_df, sortie de
    run_construction_indice()) et de la capitalisation annuelle en
    USD servant d'ancre (calculate_capi_annual_usd()).

    Étapes :
        1. Diffusion quotidienne de la capitalisation depuis la
           dernière ancre annuelle (_construire_capi_quotidien_drift)
        2. Recalcul du poids CHAQUE JOUR (_poids_quotidien_depuis_capi)
        3. Chaînage identique aux autres méthodes du projet
           (log-rendements pondérés, cumsum, Base 100) :

               r_index,t = somme_i w(i, t) * r_i,t
               Index_CAPI_t = 100 * exp( cumsum(r_index,t) )

    Returns
    -------
    dict
        {
            "capi_daily": capitalisation quotidienne diffusée (USD),
            "capi_weights_daily": poids quotidiens (recalculés chaque jour),
            "index_capi_returns": rendement log quotidien de l'indice,
            "Index_CAPI": niveau de l'indice, Base 100,
        }
    """

    capi_quotidien = _construire_capi_quotidien_drift(
        capi_annuels_usd,
        returns_df,
        list(returns_df.columns)
    )

    poids_quotidien = _poids_quotidien_depuis_capi(capi_quotidien)

    print("Première date poids : ", poids_quotidien.index.min())
    print("Dernière date poids : ", poids_quotidien.index.max())
    print("Nombre de lignes poids : ", len(poids_quotidien))
    print(poids_quotidien.head())

    # ------------------------------------------------------------
    # Les jours antérieurs à la première ancre CAPI disponible
    # n'ont pas de poids définissable : on les exclut proprement.
    # ------------------------------------------------------------

    poids_valides_mask = poids_quotidien.notna().all(axis=1)

    n_dates_exclues = (~poids_valides_mask).sum()

    if n_dates_exclues > 0:
        print(
            f"Attention : {n_dates_exclues} date(s) exclue(s) de "
            "l'indice CAPI (antérieures à la première ancre CAPI "
            "disponible)."
        )

    poids_quotidien = poids_quotidien.loc[poids_valides_mask]
    capi_quotidien = capi_quotidien.loc[poids_quotidien.index]

    returns_communs = returns_df.loc[
        poids_quotidien.index,
        poids_quotidien.columns
    ]

    index_capi_returns = (
        returns_communs
        .mul(poids_quotidien)
        .sum(axis=1)
    )

    index_capi_returns.name = "Index_CAPI_Log_Return"

    index_capi = 100 * np.exp(index_capi_returns.cumsum())
    index_capi.name = "Index_CAPI"

    return {
        "capi_daily": capi_quotidien,
        "capi_weights_daily": poids_quotidien,
        "index_capi_returns": index_capi_returns,
        "Index_CAPI": index_capi,
    }


# ================================================================
# PIPELINE COMPLET : PONDERATION CAPI + INDICE + GRAPHIQUES
# ================================================================

def run_capi_weighted_index(returns_df,
                             capi_annuels_usd=None,
                             poids_annuels=None,
                             info=None,
                             other_indices=None):
    """
    Exécute l'ensemble du pipeline de pondération par capitalisation
    boursière : capitalisation annuelle en USD (ancres), diffusion et
    rebalancement quotidiens, construction de l'indice CAPI
    (Base 100), indicateurs de concentration et graphiques.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements log quotidiens des banques, déjà nettoyés
        (sortie de run_construction_indice(), clé "returns_df").
    capi_annuels_usd : pandas.DataFrame or None
        Capitalisation annuelle en USD déjà calculée (ancres de
        rebalancement). Si None, calculate_capi_annual_usd() est
        appelée (lecture de utils.CAPI_FILE).
    poids_annuels : pandas.DataFrame or None
        Poids CAPI annuels déjà calculés (référence / reporting). Si
        None, calculés à partir de capi_annuels_usd.
    info : pandas.DataFrame or None
        Table banks_info (colonnes Ticker, Country, ...) pour
        enrichir les graphiques de composition par pays.
    other_indices : dict or None
        Dictionnaire {nom: pandas.Series} d'indices déjà construits,
        pour le graphique de comparaison.

    Returns
    -------
    dict
        Dictionnaire des objets produits, réutilisable par
        export_workbook() dans 01_Construction_Indice.py et par
        06_Comparaison_Indices.py.
    """

    print("\n" + "=" * 70)
    print("INDICE PONDERE PAR CAPITALISATION BOURSIERE (CAPI)")
    print("Rebalancement quotidien")
    print("=" * 70)

    # ============================================================
    # 1. CAPITALISATION ANNUELLE EN USD (ANCRES)
    # ============================================================

    if capi_annuels_usd is None:
        print(
            f"\nAucune capitalisation CAPI fournie : calcul à partir de "
            f"{CAPI_FILE}..."
        )
        capi_annuels_usd = calculate_capi_annual_usd()
    else:
        print("\nRéutilisation de la capitalisation CAPI déjà calculée.")

    if poids_annuels is None:
        poids_annuels = _calculer_poids(capi_annuels_usd)

    # ============================================================
    # 2. CONSTRUCTION DE L'INDICE (REBALANCEMENT QUOTIDIEN)
    # ============================================================

    print("\nConstruction de l'indice CAPI (Base 100, poids quotidiens)...")

    resultat_indice = build_capi_weighted_index(
        returns_df,
        capi_annuels_usd
    )

    poids_quotidien = resultat_indice["capi_weights_daily"]
    index_capi_returns = resultat_indice["index_capi_returns"]
    index_capi = resultat_indice["Index_CAPI"]

    print(
        f"✓ Indice CAPI construit sur {len(index_capi)} "
        "observations."
    )

    # ============================================================
    # 3. INDICATEURS DE CONCENTRATION (DERNIERE ANNEE CONNUE)
    # ============================================================

    derniers_poids = poids_annuels.iloc[-1]

    hhi_capi = float((derniers_poids ** 2).sum())

    top5_capi = float(
        derniers_poids.sort_values(ascending=False).head(5).sum()
    )

    top10_capi = float(
        derniers_poids.sort_values(ascending=False).head(10).sum()
    )

    capi_summary = pd.DataFrame({
        "Indicator": [
            "Date de la dernière ancre CAPI",
            "Nombre de banques (dernière ancre)",
            "HHI (dernière ancre)",
            "Poids Top 5 (dernière ancre)",
            "Poids Top 10 (dernière ancre)",
            "Rebalancement",
        ],
        "Value": [
            str(poids_annuels.index[-1].date())
            if hasattr(poids_annuels.index[-1], "date")
            else str(poids_annuels.index[-1]),
            int((derniers_poids > 0).sum()),
            hhi_capi,
            top5_capi,
            top10_capi,
            "Quotidien (diffusion des rendements depuis l'ancre annuelle)",
        ],
    })

    print(f"\nHHI (dernière ancre) : {hhi_capi:.4f}")
    print(f"Poids Top 5 (dernière ancre) : {top5_capi:.2%}")

    # ============================================================
    # 4. GRAPHIQUES
    # ============================================================

    print("\nCréation des graphiques de l'indice CAPI...")

    # ------------------------------------------------------------
    # 4.1 EVOLUTION DES POIDS CAPI QUOTIDIENS - TOP 10 BANQUES
    # ------------------------------------------------------------
    
    top5_tickers = (
        derniers_poids
        .sort_values(ascending=False)
        .head(4)
        .index
        .tolist()
    )

    if "ATW MC Equity" in poids_quotidien.columns:
        if "ATW MC Equity" not in top5_tickers:
            top5_tickers.append("ATW MC Equity")

    top5_tickers = [t for t in top5_tickers if t in poids_quotidien.columns]

    plt.figure(figsize=(13, 7))

    plt.gca().set_prop_cycle(
        plt.cycler(color=plt.cm.tab10.colors)
    )
    print("\nTOP 5 BANQUES RETENUES POUR LE GRAPHIQUE :")
    print(top5_tickers)
    for ticker in top5_tickers:

        plt.plot(
            poids_quotidien.index,
            poids_quotidien[ticker],
            linewidth=1.2,
            label=ticker
        )

    plt.title(
        "Évolution quotidienne des poids CAPI — Top 5 banques "
        "(dernière ancre)"
    )

    plt.xlabel("Date")
    plt.ylabel("Poids CAPI")

    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)

    save_chart("55_CAPI_Weights_Evolution")

    # ------------------------------------------------------------
    # 4.3 CONCENTRATION (HHI) DANS LE TEMPS — QUOTIDIEN
    # ------------------------------------------------------------

    hhi_dans_le_temps = (poids_quotidien ** 2).sum(axis=1)

    plt.figure(figsize=(13, 6))

    plt.plot(
        hhi_dans_le_temps.index,
        hhi_dans_le_temps.values,
        linewidth=1.2
    )

    plt.title("Concentration de l'indice CAPI (HHI) dans le temps — quotidien")
    plt.xlabel("Date")
    plt.ylabel("HHI")

    plt.grid(True, alpha=0.3)

    save_chart("57_CAPI_HHI_Evolution")

    print("\n✓ Indice CAPI (rebalancement quotidien) calculé avec succès.")

    # ============================================================
    # 5. OBJETS RENVOYES
    # ============================================================

    return {
        "capi_annuels_usd": capi_annuels_usd,
        "capi_weights_annual": poids_annuels,
        "capi_daily": resultat_indice["capi_daily"],
        "capi_weights_daily": poids_quotidien,
        "index_capi_returns": index_capi_returns,
        "Index_CAPI": index_capi,
        "capi_summary": capi_summary,
        "hhi_capi": hhi_capi,
        "top5_capi": top5_capi,
        "top10_capi": top10_capi,
    }


if __name__ == "__main__":
    calculate_capi_weights()