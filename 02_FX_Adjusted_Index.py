"""
02_FX_Adjusted_Index.py
--------------------------------------------------------------
Risque de change de l'indice bancaire africain : rendements FX,
decomposition exacte USD (local + change), DCC-GARCH bivarie
actions/change, decomposition conditionnelle du risque par
devise, nettoyage RMT + PCA -> facteur de change commun, beta de
l'indice sur ce facteur, attribution du rendement a 3 facteurs,
construction de l'indice FX-Adjusted (Base 100), comparaison
avec les indices local et USD, et calcul du score d'exposition
au risque de change normalise s_i^fx par banque.

Ce module reprend EXACTEMENT les memes calculs, formules et noms
de variables que le script d'origine pour les sections 31.1 a
31.55 (et le bloc de graphiques 53.28) : seule l'organisation du
code a change (regroupement en une fonction reutilisable).

AJOUT (mise en conformite avec la nouvelle theorie, chapitre
"Correction du biais de change et du risque systemique",
section II.3) :

    Le beta par devise beta_c, deja calcule en 31.4 comme la
    sensibilite de chaque devise au facteur de change commun
    F_t^{fx,commun}, est desormais normalise entre 0 et 1 au
    niveau pays :

        s_i^fx = (|beta_c(i)| - min_j |beta_c(j)|)
                 / (max_j |beta_c(j)| - min_j |beta_c(j)|)

    puis transpose a chaque banque i selon son pays de cotation
    c(i) (meme mapping info_country / currency_map que dans tout
    le reste du script). Le resultat est une pd.Series indexee
    EXACTEMENT comme returns_df.columns (donc alignee avec la
    matrice de covariance nettoyee Sigma_clean et avec le score
    de risque systemique s^{sys} issu du module Delta-CoVaR),
    prete a servir de vecteur s^fx dans le programme
    d'optimisation convexe bi-penalise :

        w*(lambda_sys, lambda_fx) =
            argmin_w  w' Sigma_clean w
                      + lambda_sys * s'w
                      + lambda_fx  * s^{fx}' w
            s.c.      1'w = 1,  0 <= w <= w_max

Ce module ne touche ni au classeur Excel ni au Dashboard : il se
contente de calculer les objets et de generer ses propres
graphiques (via save_chart, partage avec le module de
construction). C'est 01_Construction_Indice.py qui pilote
l'export complet et integre ces resultats.

Objets renvoyes par run_fx_adjusted_index(), directement
reutilisables par les autres modules du projet (optimisation
convexe bi-penalisee, risque systemique) :
fx_common_factor, beta_index_on_factor, attribution_index,
FX_Adjusted_Index, three_index_comparison,
fx_country_beta, fx_exposure_score_country, fx_exposure_score, etc.
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])

from arch import arch_model

from utils import save_chart


# ================================================================
# INDICE FX-ADJUSTED : PIPELINE COMPLET
# ================================================================

def run_fx_adjusted_index(
    returns_df,
    returns,
    poids_quotidien,
    fx,
    egypt_tickers,
    mondays,
    non_sunday_mask,
    info_country,
):
    """
    Execute l'integralite du pipeline lie au risque de change :
    decomposition exacte USD (local + FX), DCC-GARCH bivarie,
    decomposition conditionnelle du risque, facteur de change
    commun (RMT + PCA), score d'exposition normalise s_i^fx par
    banque, attribution a 3 facteurs, construction de l'indice
    FX-Adjusted (Base 100) et comparaison avec les indices local
    et USD.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements log USD par banque (sortie de
        run_construction_indice()).
    returns : pandas.Series
        Rendement log de l'indice (USD), post-nettoyage.
    poids_quotidien : pandas.DataFrame
        Ponderations quotidiennes des banques dans l'indice
        (rebalancees annuellement), index = dates, colonnes = tickers.
    fx : pandas.DataFrame
        Taux de change bruts (memes tickers/devises que dans le
        script d'origine).
    egypt_tickers : list
        Tickers egyptiens (regle lundi/dimanche).
    mondays : pandas.DatetimeIndex
        Dates de lundi concernees par la regle egyptienne.
    non_sunday_mask : pandas.Series
        Masque booleen excluant les dimanches.
    info_country : pandas.Series
        Mapping ticker -> pays.

    Returns
    -------
    dict
        Dictionnaire des objets produits (voir docstring du
        module), directement reutilisable par
        export_workbook() (dans 01_Construction_Indice.py) et
        par le module d'optimisation convexe bi-penalisee
        (fx_exposure_score = vecteur s^fx pret a l'emploi).
    """

    # 31.1 DECOMPOSITION EXACTE DU RENDEMENT USD (LOCAL + FX)
    # ============================================================
    #
    # Le rendement log USD d'une banque se decompose exactement en :
    #
    #   r_usd = r_local + r_fx
    #
    # car prices_common = prices * fx, et log(a*b) = log(a) + log(b).
    # On applique la meme regle Egypte (lundi vs dimanche) que pour
    # le calcul principal des rendements (section 16), afin que la
    # decomposition reste exacte, sans terme croise neglige.
    #
    # ============================================================

    print(
        "\nDecomposition exacte des rendements (local + FX)..."
    )


    def compute_log_returns_egypt_rule(price_df, egypt_tickers, mondays):
        """
        Rendements log avec la regle specifique Egypte (lundi vs dimanche).

        Reproduit exactement la logique de la section 16, generalisee a
        n'importe quelle matrice de prix (cours locaux, taux de change,
        ou devise commune) afin de garantir la decomposition exacte
        r_usd = r_local + r_fx.
        """

        returns_raw = pd.DataFrame(
            index=price_df.index,
            columns=price_df.columns,
            dtype=float,
        )

        returns_standard = np.log(
            price_df / price_df.shift(1)
        )

        returns_raw.loc[
            returns_standard.index,
            returns_standard.columns
        ] = returns_standard

        for monday in mondays:

            previous_day = monday - pd.Timedelta(days=1)

            if previous_day not in price_df.index:
                continue

            if previous_day.dayofweek != 6:
                continue

            for ticker in egypt_tickers:

                if ticker not in price_df.columns:
                    continue

                sunday_price = price_df.loc[previous_day, ticker]
                monday_price = price_df.loc[monday, ticker]

                if (
                    pd.notna(sunday_price)
                    and pd.notna(monday_price)
                    and sunday_price > 0
                    and monday_price > 0
                ):
                    returns_raw.loc[monday, ticker] = np.log(
                        monday_price / sunday_price
                    )

        return returns_raw


    # Rendements de change (FX), avec la meme regle Egypte que returns_df
    r_fx_all = compute_log_returns_egypt_rule(fx, egypt_tickers, mondays)

    r_fx_df = (
        r_fx_all
        .loc[non_sunday_mask]
        .reindex(index=returns_df.index, columns=returns_df.columns)
    )

    # Rendements "actions pures" en devise locale, obtenus par difference
    # exacte (pas d'approximation) : r_local = r_usd - r_fx
    r_local_df = returns_df - r_fx_df


    # Mapping banque -> pays (utilise comme proxy de devise), c(i) dans
    # la notation du memoire
    currency_map = info_country.to_dict()


    # Table de synthese : rendement moyen local / FX / USD par banque,
    # et correlation entre le rendement local et le rendement de change.
    fx_decomposition_summary = pd.DataFrame({
        "Rendement_Local_Moyen": r_local_df.mean(),
        "Rendement_FX_Moyen": r_fx_df.mean(),
        "Rendement_USD_Moyen": returns_df.mean(),
        "Correlation_Local_FX": r_local_df.corrwith(r_fx_df),
    })

    fx_decomposition_summary.index.name = "Ticker"

    fx_decomposition_summary = (
        fx_decomposition_summary
        .reset_index()
    )

    fx_decomposition_summary["Country"] = (
        fx_decomposition_summary["Ticker"].map(currency_map)
    )


    # Controle : la reconstruction locale + FX doit reproduire exactement
    # les rendements USD deja calcules en section 16.
    _reconstruction_error = (
        (r_local_df + r_fx_df) - returns_df
    ).abs().max().max()

    print(
        "Erreur maximale de reconstruction (local + FX vs USD) : "
        f"{_reconstruction_error:.2e}"
    )

    # ============================================================
    # EXEMPLE REEL DE DECOMPOSITION (pour le rapport)
    # ============================================================

    _common_dates = (
        r_local_df.dropna(how="all").index
        .intersection(r_fx_df.dropna(how="all").index)
        .intersection(returns_df.dropna(how="all").index)
    )

    fx_impact = r_fx_df.abs().mean(axis=1)

    example_date = fx_impact.idxmax()

    # Banque imposée
    mandatory_ticker = "ATW MC Equity"

    selected_tickers = []

    # Vérifie qu'elle existe et qu'elle possède des données
    if (
        mandatory_ticker in returns_df.columns
        and pd.notna(r_local_df.loc[example_date, mandatory_ticker])
        and pd.notna(r_fx_df.loc[example_date, mandatory_ticker])
        and pd.notna(returns_df.loc[example_date, mandatory_ticker])
    ):
        selected_tickers.append(mandatory_ticker)
    else:
        print(f"Attention : {mandatory_ticker} indisponible pour la date {example_date.date()}.")

    preferred_tickers = [
        "ATW MC Equity",
        "COMI EY Equity",
        "SBK SJ Equity"
    ]

    for ticker in preferred_tickers:
        if ticker in returns_df.columns:
            if (
                pd.notna(r_local_df.loc[example_date,ticker])
                and pd.notna(r_fx_df.loc[example_date,ticker])
                and pd.notna(returns_df.loc[example_date,ticker])
            ):
                if ticker not in selected_tickers:
                    selected_tickers.append(ticker)

    example_rows = []

    for ticker in selected_tickers:

        example_rows.append({
            "Date": example_date.date(),
            "Ticker": ticker,
            "Country": info_country.get(ticker, ""),
            "Weight (%)": poids_quotidien.loc[example_date, ticker] * 100,
            "Local Return (%)": r_local_df.loc[example_date, ticker] * 100,
            "FX Return (%)": r_fx_df.loc[example_date, ticker] * 100,
            "USD Return (%)": returns_df.loc[example_date, ticker] * 100,
            "Reconstruction Error": (
                r_local_df.loc[example_date, ticker]
                + r_fx_df.loc[example_date, ticker]
                - returns_df.loc[example_date, ticker]
            )
        })

    fx_real_example = pd.DataFrame(example_rows)

    print("\nExemple réel de décomposition :")
    print(fx_real_example)

    # ============================================================
    # 31.2 DCC-GARCH BIVARIE ACTIONS / CHANGE (PAR DEVISE)
    # ============================================================
    #
    # Les paramètres DCC (a,b) ne sont pas imposés.
    # Ils sont estimés par maximum de vraisemblance conditionnelle
    # (QMLE) sous les contraintes :
    #
    # a >=0, b>=0, a+b<1
    #
    # conformément à Engle (2002).
    #
    # ============================================================

    _fx_marginals_cache = {}


    def fit_garch_marginals(series: pd.Series):
        """GARCH(1,1) univarie ; retourne vol conditionnelle et residus standardises."""

        am = arch_model(
            series.dropna() * 100,
            vol="Garch",
            p=1,
            q=1,
            dist="t",
        )

        try:
            res = am.fit(
                disp="off",
                options={
                    "maxiter": 2000
                }
            )

        except Exception:

            res = am.fit(
                disp="off",
                method="powell"
            )

        sigma_t = res.conditional_volatility / 100

        z_t = (
            series.dropna() - res.params["mu"] / 100
        ) / sigma_t

        return sigma_t, z_t, res


    def dcc_correlation(z1: pd.Series, z2: pd.Series):
        """
        Estimation DCC(1,1) par maximum de vraisemblance.

        Q_t = (1-a-b)Qbar + a z_{t-1}z'_{t-1} + b Q_{t-1}

        avec contraintes :
            a >= 0
            b >= 0
            a+b < 1
        """

        idx = z1.index.intersection(z2.index)

        Z = np.column_stack([
            z1.loc[idx].values,
            z2.loc[idx].values
        ])

        T = len(Z)

        Qbar = np.cov(Z.T)

        def dcc_loglik(params):

            a, b = params

            # contraintes DCC
            if (
                a < 0
                or b < 0
                or a + b >= 0.999
            ):
                return 1e10


            Q_t = Qbar.copy()

            loglik = 0


            for t in range(1, T):

                z_prev = Z[t-1].reshape(-1,1)


                Q_t = (
                    (1-a-b)*Qbar
                    +
                    a*(z_prev @ z_prev.T)
                    +
                    b*Q_t
                )


                d = np.sqrt(
                    np.diag(Q_t)
                )

                R_t = (
                    Q_t
                    /
                    np.outer(d,d)
                )


                det_R = np.linalg.det(R_t)


                if det_R <= 0:
                    return 1e10


                inv_R = np.linalg.inv(R_t)


                z = Z[t].reshape(-1,1)


                loglik += (
                    np.log(det_R)
                    +
                    z.T @ inv_R @ z
                )


            return float(loglik)



        # valeurs initiales
        x0 = [
            0.02,
            0.95
        ]


        result = minimize(
            dcc_loglik,
            x0,
            method="SLSQP",
            bounds=[
                (1e-6,0.999),
                (1e-6,0.999)
            ],
            constraints={
                "type":"ineq",
                "fun":lambda x:0.999-x.sum()
            }
        )


        if result.success:

            a_est, b_est = result.x

        else:

            print(
                "Attention estimation DCC échouée, "
                "utilisation des valeurs initiales"
            )

            a_est, b_est = x0



        print(
            f"Paramètres DCC estimés : "
            f"a={a_est:.4f}, b={b_est:.4f}"
        )


        # Reconstruction des corrélations conditionnelles

        Q_t = Qbar.copy()

        rho_series = np.zeros(T)


        for t in range(1,T):

            z_prev = Z[t-1].reshape(-1,1)


            Q_t = (
                (1-a_est-b_est)*Qbar
                +
                a_est*(z_prev@z_prev.T)
                +
                b_est*Q_t
            )


            d=np.sqrt(
                np.diag(Q_t)
            )


            R_t = (
                Q_t
                /
                np.outer(d,d)
            )


            rho_series[t]=R_t[0,1]



        return pd.Series(
            rho_series,
            index=idx,
            name="rho_loc_fx_t"
        )


    def risk_decomposition_conditional(bank: str,
                                        returns_local_log: pd.DataFrame,
                                        returns_fx_log: pd.DataFrame,
                                        currency_map: dict) -> pd.DataFrame:
        """Variance conditionnelle de r_USD et part imputable au change, jour par jour."""

        ccy = currency_map[bank]

        r_loc = returns_local_log[bank].dropna()

        if ccy not in _fx_marginals_cache:
            r_fx_full = returns_fx_log[ccy].dropna()
            _fx_marginals_cache[ccy] = fit_garch_marginals(r_fx_full)

        sigma_fx, z_fx, _ = _fx_marginals_cache[ccy]
        sigma_loc, z_loc, _ = fit_garch_marginals(r_loc)

        rho_t = dcc_correlation(z_loc, z_fx)

        idx = (
            sigma_loc.index
            .intersection(sigma_fx.index)
            .intersection(rho_t.index)
        )

        var_loc = sigma_loc.loc[idx] ** 2
        var_fx = sigma_fx.loc[idx] ** 2
        cov_t = rho_t.loc[idx] * sigma_loc.loc[idx] * sigma_fx.loc[idx]
        var_usd = var_loc + var_fx + 2 * cov_t

        return pd.DataFrame({
            "var_local": var_loc,
            "var_fx": var_fx,
            "cov_local_fx": cov_t,
            "var_usd": var_usd,
            "part_change_pct": (var_fx + 2 * cov_t) / var_usd,
            "rho_local_fx": rho_t.loc[idx],
        })


    # ============================================================
    # 31.3 DECOMPOSITION CONDITIONNELLE DU RISQUE, PAR BANQUE
    # ============================================================
    #
    # On regroupe d'abord les rendements de change par pays (= devise),
    # en moyennant les banques d'un meme pays (elles partagent en principe
    # le meme taux de change). On applique ensuite le DCC-GARCH ci-dessus
    # a chaque banque disposant d'un historique suffisant.
    #
    # ============================================================

    print(
        "\nDecomposition conditionnelle du risque (DCC-GARCH, par banque)..."
    )
    print(
        "(cette etape peut prendre plusieurs minutes selon le nombre de banques)"
    )


    _tickers_by_country = (
        info_country
        .reset_index()
        .groupby("Country")["Ticker"]
        .apply(list)
        .to_dict()
    )

    _fx_by_country_data = {}

    for _country, _tickers in _tickers_by_country.items():

        _tickers_ok = [t for t in _tickers if t in r_fx_df.columns]

        if not _tickers_ok:
            continue

        _fx_by_country_data[_country] = r_fx_df[_tickers_ok].mean(axis=1)

    fx_by_country = pd.DataFrame(
        _fx_by_country_data,
        index=r_fx_df.index,
    )


    MIN_OBS_DCC_GARCH = 250

    conditional_risk_daily = {}
    conditional_risk_summary_rows = []

    for bank in returns_df.columns:

        ccy = currency_map.get(bank)

        if ccy is None or ccy not in fx_by_country.columns:
            print(f"  ! {bank} : devise inconnue, ignoree.")
            continue

        if r_local_df[bank].dropna().shape[0] < MIN_OBS_DCC_GARCH:
            print(f"  ! {bank} : historique insuffisant, ignoree.")
            continue

        try:
            table = risk_decomposition_conditional(
                bank,
                r_local_df,
                fx_by_country,
                currency_map,
            )
        except Exception as e:
            print(f"  ! {bank} : echec DCC-GARCH ({e}).")
            continue

        conditional_risk_daily[bank] = table

        conditional_risk_summary_rows.append({
            "Ticker": bank,
            "Country": ccy,
            "Var_Local_Moyenne": table["var_local"].mean(),
            "Var_FX_Moyenne": table["var_fx"].mean(),
            "Cov_Local_FX_Moyenne": table["cov_local_fx"].mean(),
            "Var_USD_Moyenne": table["var_usd"].mean(),
            "Part_Change_Moyenne (%)": table["part_change_pct"].mean() * 100,
            "Rho_Local_FX_Moyen": table["rho_local_fx"].mean(),
        })

    conditional_risk_summary = pd.DataFrame(
        conditional_risk_summary_rows
    )

    print(
        f"\nDCC-GARCH estime avec succes pour "
        f"{len(conditional_risk_daily)} / {len(returns_df.columns)} banques."
    )


    # ============================================================
    # 31.4 FACTEUR DE CHANGE COMMUN (NETTOYAGE RMT + PCA)
    # ============================================================
    #
    # Nettoyage RMT (Laloux-Cizeau-Bouchaud-Potters) de la matrice de
    # correlation des rendements de change par pays, puis extraction
    # du facteur de change commun regional = 1ere composante principale
    # significative (au-dessus du bord superieur de Marchenko-Pastur).
    #
    # ============================================================

    print(
        "\nExtraction du facteur de change commun (RMT + PCA)..."
    )


    def marchenko_pastur_bounds(T: int, N: int, sigma2: float = 1.0):

        q = N / T

        lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
        lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2

        return lam_minus, lam_plus


    def rmt_clean_correlation(returns_fx: pd.DataFrame) -> np.ndarray:
        """
        Nettoyage RMT (Laloux-Cizeau-Bouchaud-Potters) de la matrice de correlation
        des rendements de change : les valeurs propres dans le bulk de Marchenko-Pastur
        sont considerees comme du bruit et remplacees par leur moyenne.
        """

        X = returns_fx.dropna()

        T, N = X.shape

        C = np.corrcoef(X.values.T)

        eigvals, eigvecs = np.linalg.eigh(C)

        lam_minus, lam_plus = marchenko_pastur_bounds(T, N)

        noise_mask = (eigvals >= lam_minus) & (eigvals <= lam_plus)

        avg_noise_eigval = (
            eigvals[noise_mask].mean()
            if noise_mask.any()
            else 0.0
        )

        eigvals_clean = np.where(noise_mask, avg_noise_eigval, eigvals)

        C_clean = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T

        d = np.sqrt(np.diag(C_clean))
        C_clean = C_clean / np.outer(d, d)

        return C_clean, eigvals, eigvecs, lam_plus


    def common_fx_factor(returns_fx: pd.DataFrame):
        """
        Facteur de change commun = 1ere composante principale de la matrice
        de correlation FX nettoyee par RMT. Retourne le facteur (serie temporelle)
        et les betas de chaque devise sur ce facteur (beta_c dans la notation
        du memoire, section II.3).
        """

        C_clean, eigvals, eigvecs, lam_plus = rmt_clean_correlation(returns_fx)

        signal_idx = np.where(eigvals > lam_plus)[0]

        if len(signal_idx) == 0:

            print(
                "Aucune composante RMT significative. "
                "Utilisation de la première composante principale."
            )

            top_idx = np.argmax(eigvals)

        else:

            top_idx = signal_idx[
                np.argmax(
                    eigvals[signal_idx]
                )
            ]

        loadings = eigvecs[:, top_idx]
        loadings = loadings / np.linalg.norm(loadings)

        X = returns_fx.dropna()

        factor = pd.Series(
            X.values @ loadings,
            index=X.index,
            name="fx_common_factor",
        )

        betas = {}

        for ccy in returns_fx.columns:

            y = X[ccy].values

            beta = (
                np.cov(y, factor.values)[0, 1] / np.var(factor.values)
            )

            betas[ccy] = beta

        return factor, pd.Series(betas, name="beta_on_fx_factor"), C_clean


    if fx_by_country.shape[1] >= 2:

        try:
            fx_common_factor, fx_betas, fx_corr_clean = common_fx_factor(fx_by_country)
        except ValueError as e:
            print(f"  ! Facteur de change commun : {e}")
            fx_common_factor, fx_betas = None, None

    else:
        print(
            "  ! Une seule devise detectee : pas de facteur commun a extraire."
        )
        fx_common_factor, fx_betas = None, None

    # fx_betas est le beta_c de la section II.3 du memoire, indexe par
    # pays. On le conserve tel quel sous ce nom explicite pour la suite
    # du pipeline (ERC bi-penalise notamment). fx_country_beta est un
    # alias explicite du meme objet, utilise plus bas pour le calcul du
    # score d'exposition normalise (et par le module d'optimisation).
    fx_country_beta = fx_betas

    print("\nDEBUG FX BETA COUNTRY")
    print(fx_country_beta)

    if fx_country_beta is not None:
        print(
            "Nombre de devises avec beta FX :",
            len(fx_country_beta)
        )


    # ============================================================
    # 31.45 SCORE D'EXPOSITION AU RISQUE DE CHANGE NORMALISE s_i^fx
    # (NOUVELLE THEORIE, SECTION II.3, EQUATION eq:score_fx_normalise)
    # ============================================================
    #
    # Le beta par pays beta_c est normalise entre 0 et 1 :
    #
    #   s_i^fx = (|beta_c(i)| - min_j |beta_c(j)|)
    #            / (max_j |beta_c(j)| - min_j |beta_c(j)|)
    #
    # puis transpose a chaque banque i selon son pays de cotation c(i)
    # (mapping currency_map, identique a tout le reste du script). Le
    # resultat est une pd.Series indexee exactement comme
    # returns_df.columns : c'est directement le vecteur s^fx a passer
    # au programme d'optimisation convexe bi-penalise (eq. ERC
    # bi-penalise), aligne avec Sigma_clean et avec le score de risque
    # systemique s^sys (module Delta-CoVaR).
    #
    # ============================================================

    print(
        "\nCalcul du score d'exposition au risque de change normalise (s_i^fx)..."
    )


    def normalized_fx_exposure_score(beta_by_country: pd.Series) -> pd.Series:
        """
        Normalise |beta_c| entre 0 et 1 (min-max), au niveau pays.

        Cas degenere (une seule devise, ou tous les |beta_c| egaux) :
        min = max, la formule est indeterminee (0/0) ; on retourne
        alors un score neutre de 0.5 pour chaque pays, plutot que de
        laisser NaN se propager dans l'optimisation convexe en aval.
        """

        abs_beta = beta_by_country.abs()

        beta_min = abs_beta.min()
        beta_max = abs_beta.max()

        spread = beta_max - beta_min

        if spread == 0 or pd.isna(spread):
            return pd.Series(
                0.5,
                index=abs_beta.index,
                name="fx_exposure_score",
            )

        score = (abs_beta - beta_min) / spread

        score.name = "fx_exposure_score"

        return score


    if (
        fx_country_beta is not None
        and len(fx_country_beta) > 0
    ):

        # Score normalise, au niveau pays : s_c^fx, c in {pays du panier}
        fx_exposure_score_country = normalized_fx_exposure_score(
            fx_country_beta
        )

        # Transposition a chaque banque i selon son pays de cotation c(i)
        fx_exposure_score = (
            pd.Series(currency_map, name="Country")
            .map(fx_exposure_score_country)
            .reindex(returns_df.columns)
        )

        fx_exposure_score = (
            fx_exposure_score
            .fillna(
                fx_exposure_score_country.mean()
            )
        )

        fx_exposure_score.name = "fx_exposure_score"

        _n_missing_score = fx_exposure_score.isna().sum()

        if _n_missing_score > 0:
            print(
                f"  ! {_n_missing_score} banque(s) sans score s^fx "
                "(pays absent de fx_country_beta) : verifier info_country."
            )

        print(
            "Score d'exposition FX (s^fx) — min / mediane / max : "
            f"{fx_exposure_score.min():.4f} / "
            f"{fx_exposure_score.median():.4f} / "
            f"{fx_exposure_score.max():.4f}"
        )

    else:
        fx_exposure_score_country = None
        fx_exposure_score = None
        print(
            "  ! Pas de beta_c disponible : score s^fx non calcule."
        )


    # ============================================================
    # 31.5 ATTRIBUTION A 3 FACTEURS DE L'INDICE
    # ============================================================
    #
    # Decompose le rendement log USD de l'indice en 3 composantes :
    #   - performance actions pure (devise locale)
    #   - effet de change systematique (facteur commun regional)
    #   - effet de change idiosyncratique residuel
    #
    # ============================================================


    def three_factor_attribution(index_usd_log_ret: pd.Series,
                                  index_local_log_ret: pd.Series,
                                  fx_common_factor: pd.Series,
                                  beta_index_on_factor: float) -> pd.DataFrame:
        """
        Decompose le rendement log de l'indice USD en 3 composantes :
        - performance actions pure (devise locale)
        - effet de change systematique (facteur commun regional)
        - effet de change idiosyncratique residuel
        """

        idx = index_usd_log_ret.index.intersection(index_local_log_ret.index)
        idx = idx.intersection(fx_common_factor.index)

        r_usd = index_usd_log_ret.loc[idx]
        r_local = index_local_log_ret.loc[idx]

        systematic_fx = beta_index_on_factor * fx_common_factor.loc[idx]
        idiosyncratic_fx = (r_usd - r_local) - systematic_fx

        return pd.DataFrame({
            "rendement_actions_pur": r_local,
            "effet_change_systematique": systematic_fx,
            "effet_change_idiosyncratique": idiosyncratic_fx,
            "rendement_usd_total": r_usd,
        })

    poids_quotidien_local = poids_quotidien.reindex(
        index=r_local_df.index,
        columns=r_local_df.columns
    )

    index_local_returns = (
        r_local_df
        .mul(poids_quotidien_local)
        .sum(axis=1)
        .dropna()
    )

    if fx_common_factor is not None:

        _idx_common = (
            returns.index
            .intersection(index_local_returns.index)
            .intersection(fx_common_factor.index)
        )

        _index_fx_effect = (
            returns.loc[_idx_common] - index_local_returns.loc[_idx_common]
        )

        beta_index_on_factor = (
            np.cov(
                _index_fx_effect.values,
                fx_common_factor.loc[_idx_common].values,
            )[0, 1]
            / np.var(fx_common_factor.loc[_idx_common].values)
        )

        attribution_index = three_factor_attribution(
            returns,
            index_local_returns,
            fx_common_factor,
            beta_index_on_factor,
        )

        print(
            "\nBeta de l'indice sur le facteur de change commun : "
            f"{beta_index_on_factor:.4f}"
        )

    else:
        attribution_index = None
        beta_index_on_factor = np.nan


    # ============================================================
    # 31.55 INDICE FX-ADJUSTED ET COMPARAISON DES TROIS INDICES
    # (INDICE LOCAL / INDICE USD / INDICE FX-ADJUSTED)
    # ============================================================
    #
    # On reconstruit ici un veritable indice en niveau (Base 100),
    # et non plus seulement une decomposition des rendements, en
    # reutilisant exactement la meme logique de chainage log-additif
    # que pour les autres indices du script :
    #
    #   Indice_t = 100 * exp(cumsum(rendement_log))
    #
    # Le rendement ajuste retire l'effet de change systematique
    # (facteur commun regional) du rendement total de l'indice USD :
    #
    #   adjusted_log_return = rendement_usd_total
    #                        - effet_change_systematique
    #
    # ============================================================

    print(
        "\nConstruction de l'indice FX-Adjusted..."
    )

    if attribution_index is not None:

        # ------------------------------------------------------------
        # Rendement ajuste de l'effet de change systematique
        # ------------------------------------------------------------

        adjusted_log_return = (
            attribution_index["rendement_usd_total"]
            -
            attribution_index["effet_change_systematique"]
        )

        adjusted_log_return.name = "Adjusted_Log_Return"

        # ------------------------------------------------------------
        # Indice FX-Adjusted (Base 100)
        # ------------------------------------------------------------

        FX_Adjusted_Index = (
            100
            *
            np.exp(
                adjusted_log_return.cumsum()
            )
        )

        FX_Adjusted_Index.name = "FX_Adjusted_Index"

        # ------------------------------------------------------------
        # Indice USD rebase sur la fenetre d'attribution commune
        # (meme logique Base 100 que le reste du script)
        # ------------------------------------------------------------

        index_usd_common_returns = (
            attribution_index["rendement_usd_total"]
        )

        Index_USD_ThreeWay = (
            100
            *
            np.exp(
                index_usd_common_returns.cumsum()
            )
        )

        Index_USD_ThreeWay.name = "Index_USD"

        # ------------------------------------------------------------
        # Indice local (devise locale, Base 100)
        # ------------------------------------------------------------

        index_local_common_returns = (
            attribution_index["rendement_actions_pur"]
        )

        Index_Local_ThreeWay = (
            100
            *
            np.exp(
                index_local_common_returns.cumsum()
            )
        )

        Index_Local_ThreeWay.name = "Index_Local"

        # ------------------------------------------------------------
        # Tableau de comparaison des trois indices
        # ------------------------------------------------------------

        three_index_comparison = pd.DataFrame({
            "Index_Local": Index_Local_ThreeWay,
            "Index_USD": Index_USD_ThreeWay,
            "Index_FX_Adjusted": FX_Adjusted_Index,
        })

        three_index_comparison = (
            three_index_comparison
            .dropna()
        )

        # ------------------------------------------------------------
        # Differences absolues
        # ------------------------------------------------------------

        three_index_comparison["Diff_USD_vs_FXAdjusted"] = (
            three_index_comparison["Index_FX_Adjusted"]
            -
            three_index_comparison["Index_USD"]
        )

        three_index_comparison["Diff_Local_vs_FXAdjusted"] = (
            three_index_comparison["Index_FX_Adjusted"]
            -
            three_index_comparison["Index_Local"]
        )

        # ------------------------------------------------------------
        # Differences relatives (%)
        # ------------------------------------------------------------

        three_index_comparison["Diff_USD_vs_FXAdjusted_Pct"] = (
            three_index_comparison["Diff_USD_vs_FXAdjusted"]
            /
            three_index_comparison["Index_USD"]
        ) * 100

        three_index_comparison["Diff_Local_vs_FXAdjusted_Pct"] = (
            three_index_comparison["Diff_Local_vs_FXAdjusted"]
            /
            three_index_comparison["Index_Local"]
        ) * 100

        # ------------------------------------------------------------
        # Rendements des trois indices (log-rendements deja calcules)
        # ------------------------------------------------------------

        three_index_returns = pd.DataFrame({
            "Log_Return_Local": index_local_common_returns,
            "Log_Return_USD": index_usd_common_returns,
            "Log_Return_FX_Adjusted": adjusted_log_return,
        }).dropna()

        # ------------------------------------------------------------
        # Correlations
        # ------------------------------------------------------------

        three_index_correlations = (
            three_index_returns.corr()
        )

        # ------------------------------------------------------------
        # Rendements moyens et volatilites (quotidiens + annualises)
        # ------------------------------------------------------------

        three_index_mean_returns = (
            three_index_returns.mean()
        )

        three_index_volatility = (
            three_index_returns.std()
        )

        three_index_mean_returns_annualized = (
            three_index_mean_returns * 252
        )

        three_index_volatility_annualized = (
            three_index_volatility * np.sqrt(252)
        )

        # ------------------------------------------------------------
        # Tracking error : indice USD vs indice FX-Adjusted
        # ------------------------------------------------------------

        tracking_diff = (
            three_index_returns["Log_Return_USD"]
            -
            three_index_returns["Log_Return_FX_Adjusted"]
        )

        tracking_error_daily = tracking_diff.std()

        tracking_error_annualized = (
            tracking_error_daily * np.sqrt(252)
        )

        # ------------------------------------------------------------
        # Impact moyen du facteur de change (effet systematique)
        # ------------------------------------------------------------

        mean_fx_systematic_effect = (
            attribution_index["effet_change_systematique"].mean()
        )

        corr_usd_fxadjusted = (
            three_index_returns["Log_Return_USD"]
            .corr(
                three_index_returns["Log_Return_FX_Adjusted"]
            )
        )

        corr_local_fxadjusted = (
            three_index_returns["Log_Return_Local"]
            .corr(
                three_index_returns["Log_Return_FX_Adjusted"]
            )
        )

        # ------------------------------------------------------------
        # Tableau de synthese
        # ------------------------------------------------------------

        three_index_comparison_summary = pd.DataFrame({

            "Indicator": [

                "Rendement moyen quotidien - Local",
                "Rendement moyen quotidien - USD",
                "Rendement moyen quotidien - FX-Adjusted",
                "Rendement moyen annualise - Local",
                "Rendement moyen annualise - USD",
                "Rendement moyen annualise - FX-Adjusted",
                "Volatilite annualisee - Local",
                "Volatilite annualisee - USD",
                "Volatilite annualisee - FX-Adjusted",
                "Correlation USD / FX-Adjusted",
                "Correlation Local / FX-Adjusted",
                "Tracking error annualise (USD vs FX-Adjusted)",
                "Impact moyen du facteur de change (quotidien)",

            ],

            "Value": [

                three_index_mean_returns["Log_Return_Local"],
                three_index_mean_returns["Log_Return_USD"],
                three_index_mean_returns["Log_Return_FX_Adjusted"],
                three_index_mean_returns_annualized["Log_Return_Local"],
                three_index_mean_returns_annualized["Log_Return_USD"],
                three_index_mean_returns_annualized["Log_Return_FX_Adjusted"],
                three_index_volatility_annualized["Log_Return_Local"],
                three_index_volatility_annualized["Log_Return_USD"],
                three_index_volatility_annualized["Log_Return_FX_Adjusted"],
                corr_usd_fxadjusted,
                corr_local_fxadjusted,
                tracking_error_annualized,
                mean_fx_systematic_effect,

            ]

        })

        print(
            "\nTracking error annualise (USD vs FX-Adjusted) : "
            f"{tracking_error_annualized:.6f}"
        )

        print(
            "Correlation USD / FX-Adjusted : "
            f"{corr_usd_fxadjusted:.6f}"
        )

    else:

        adjusted_log_return = None
        FX_Adjusted_Index = None
        Index_USD_ThreeWay = None
        Index_Local_ThreeWay = None
        index_local_common_returns = None
        index_usd_common_returns = None
        three_index_comparison = None
        three_index_returns = None
        three_index_correlations = None
        three_index_comparison_summary = None
        three_index_mean_returns = None
        three_index_volatility_annualized = None
        tracking_error_annualized = np.nan
        corr_usd_fxadjusted = np.nan
        corr_local_fxadjusted = np.nan
        mean_fx_systematic_effect = np.nan

        print(
            "  ! Pas de facteur de change commun disponible : "
            "l'indice FX-Adjusted n'est pas construit."
        )


    # ============================================================
    # 53.28 GRAPHIQUES : INDICE FX-ADJUSTED ET COMPARAISON
    # DES TROIS INDICES (LOCAL / USD / FX-ADJUSTED)
    # ============================================================

    if three_index_comparison is not None:

        # --------------------------------------------------------
        # 53.28.3 EVOLUTION DU FACTEUR FX COMMUN
        # --------------------------------------------------------

        plt.figure(
            figsize=(13, 6)
        )

        plt.plot(
            fx_common_factor.index,
            fx_common_factor.values,
            linewidth=1.2,
            color="darkred"
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        plt.title(
            "Évolution du facteur de change commun (RMT + PCA)"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            "Facteur de change commun"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        save_chart(
            "29_Evolution_Facteur_FX_Commun"
        )

        # --------------------------------------------------------
        # 53.28.4 EVOLUTION DE L'EFFET DE CHANGE SYSTEMATIQUE
        # --------------------------------------------------------

        plt.figure(
            figsize=(13, 6)
        )

        plt.plot(
            attribution_index.index,
            attribution_index[
                "effet_change_systematique"
            ],
            linewidth=1.2,
            color="darkorange"
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        plt.title(
            "Évolution de l'effet de change systématique de l'indice"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            "Effet de change systématique (log-rendement)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        save_chart(
            "30_Evolution_Effet_Change_Systematique"
        )

    # --------------------------------------------------------
    # 53.28.6 SCORE D'EXPOSITION FX NORMALISE PAR BANQUE (s^fx)
    # --------------------------------------------------------

    if fx_exposure_score is not None:

        _plot_data = fx_exposure_score.dropna().sort_values()

        plt.figure(
            figsize=(13, max(6, 0.28 * len(_plot_data)))
        )

        plt.barh(
            _plot_data.index,
            _plot_data.values,
            color=PRIMARY_COLOR,
        )

        plt.title(
            "Score d'exposition au risque de change normalise (s^fx) par banque"
        )

        plt.xlabel(
            "s_i^fx  (0 = devise la moins exposee au facteur commun, "
            "1 = la plus exposee)"
        )

        plt.grid(
            True,
            axis="x",
            alpha=0.3
        )

        plt.tight_layout()

        save_chart(
            "32_Score_Exposition_FX_Par_Banque"
        )


    # ============================================================


    # Objets renvoyes pour reutilisation par les autres modules
    return {
        "r_fx_df": r_fx_df,
        "r_local_df": r_local_df,
        "reconstruction_error": _reconstruction_error,
        "fx_decomposition_summary": fx_decomposition_summary,
        "fx_real_example": fx_real_example,
        "fx_by_country": fx_by_country,
        "fx_common_factor": fx_common_factor,
        "beta_index_on_factor": beta_index_on_factor,
        "fx_betas": fx_betas,
        "fx_country_beta": fx_country_beta,
        "fx_exposure_score_country": fx_exposure_score_country,
        "fx_exposure_score": fx_exposure_score,
        "conditional_risk_summary": conditional_risk_summary,
        "attribution_index": attribution_index,
        "index_local_returns": index_local_returns,
        "index_local_common_returns": index_local_common_returns,
        "index_usd_common_returns": index_usd_common_returns,
        "adjusted_log_return": adjusted_log_return,
        "FX_Adjusted_Index": FX_Adjusted_Index,
        "Index_USD_ThreeWay": Index_USD_ThreeWay,
        "Index_Local_ThreeWay": Index_Local_ThreeWay,
        "three_index_comparison": three_index_comparison,
        "three_index_returns": three_index_returns,
        "three_index_correlations": three_index_correlations,
        "three_index_comparison_summary": three_index_comparison_summary,
        "three_index_mean_returns": three_index_mean_returns,
        "three_index_volatility_annualized": three_index_volatility_annualized,
        "tracking_error_annualized": tracking_error_annualized,
        "corr_usd_fxadjusted": corr_usd_fxadjusted,
        "corr_local_fxadjusted": corr_local_fxadjusted,
        "mean_fx_systematic_effect": mean_fx_systematic_effect,
    }