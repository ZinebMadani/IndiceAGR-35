# AGR35 — Indice bancaire panafricain ajusté au risque systémique

Construction d'un indice de 35 banques réparties sur 14 pays africains, par
**optimisation convexe bi-pénalisée** intégrant le risque systémique
(ΔCoVaR) et le risque de change.

Projet réalisé dans le cadre d'un stage à **Attijari Global Research (AGR)**,
division finance de marché (INSEA, filière Actuariat/Finance, 2025-2026).

---

## 1. Objectif

Construire un indice bancaire panafricain dont les pondérations tiennent
compte, directement dans le programme d'optimisation, de deux sources de
risque propres aux marchés africains :

- le **risque systémique** de chaque banque (score issu du ΔCoVaR) ;
- le **risque de change** (exposition à un facteur de change commun).

L'indice est comparé à trois portefeuilles de référence : capitalisation
boursière, variance minimale et ERC (Equal Risk Contribution).

## 2. Méthodologie

Les poids sont obtenus en résolvant :

```
w*(λ_sys, λ_fx) = argmin_w   wᵀ Σ_clean w  +  λ_sys · sᵀw  +  λ_fx · s^fxᵀ w
                  s.c.       1ᵀw = 1,   0 ≤ w ≤ w_max
```

| Élément | Description |
|---|---|
| `Σ_clean` | Matrice de covariance nettoyée par la théorie des matrices aléatoires (RMT, Marchenko-Pastur) |
| `s` | Score de risque systémique normalisé, issu du ΔCoVaR |
| `s^fx` | Score d'exposition au change normalisé, issu d'un facteur de change commun (DCC-GARCH, RMT + ACP) |
| `w_max` | Plafond de pondération (15 % par défaut) |
| `λ_sys`, `λ_fx` | Intensités de pénalisation (1 par défaut) |

Les deux termes de pénalité sont mis à l'échelle de la variance du portefeuille
équipondéré, de sorte qu'une valeur de λ de l'ordre de 1 correspond à une
pénalisation du même ordre de grandeur que le terme de risque.

À `λ_sys = λ_fx = 0`, le programme se réduit au portefeuille à variance minimale.
Le programme est strictement convexe : la solution existe et est unique.

**Portefeuilles de référence**

- **Market Cap** : pondération par capitalisation (poids fournis en entrée) ;
- **Min-Variance** : programme ci-dessus à `(0, 0)` ;
- **ERC** : programme log-barrière de Spinu (2013), résolu par la méthode de
  Newton, sans plafond `w_max`.

## 3. Structure du projet

```
.
├── main.py                        # Orchestration et export Excel
├── utils.py                       # Fonctions communes (save_chart)
├── 01_Construction_Indice.py      # Indice Base 100, tests statistiques
├── 02_FX_Adjusted_Index.py        # Facteur de change commun, score s^fx
├── 03_RMT_HRP_Index.py            # Nettoyage RMT de la covariance
├── 04_Systemic_Risk.py            # ΔCoVaR, VaR, ES, score s
├── 05_Convex_Optimization.py      # Optimisation bi-pénalisée, Min-Var, ERC
├── 06_Comparaison_Indices.py      # Comparaison des indices
└── 07_Calcul_Poids_CAPI.py        # Pondération par capitalisation
```

Les modules sont exécutés dans l'ordre par `main.py`, chacun transmettant ses
résultats au suivant (par exemple Σ_clean, `s` et `s^fx` vers l'optimisation).

| Module | Rôle |
|---|---|
| `01` | Rendements logarithmiques, indice Base 100, statistiques descriptives, tests ADF, Phillips-Perron, Jarque-Bera, Ljung-Box, effet ARCH, GARCH, HHI |
| `02` | Facteur de change commun et score d'exposition `s^fx` par banque |
| `03` | Filtrage du bruit de la matrice de covariance (RMT) |
| `04` | ΔCoVaR par banque, VaR, Expected Shortfall, score `s` |
| `05` | Portefeuilles Min-Variance, ERC et bi-pénalisé, frontières risque/exposition, indicateurs |
| `06` | Comparaison de l'indice par capitalisation et des indices optimisés |
| `07` | Poids de capitalisation et évolution de la concentration |

## 4. Installation

Python 3.10 ou supérieur recommandé.

```bash
git clone <url-du-depot>
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install numpy pandas matplotlib scipy statsmodels arch cvxpy osqp openpyxl
```

> Les bibliothèques ci-dessus correspondent aux usages décrits (optimisation
> `cvxpy`/OSQP, tests statistiques, GARCH). Compléter ou ajuster selon le
> fichier `requirements.txt` du projet.

## 5. Données d'entrée

- **Matrice des cours journaliers** des 35 banques (tickers Bloomberg, par
  exemple `ABBL BG Equity`) ;
- **Fichier d'informations** sur les banques (pays, poids de capitalisation) ;
- **Séries de change** utilisées pour le facteur de change commun.

> Placer les fichiers dans le dossier prévu par `main.py` et adapter les chemins
> en tête de ce fichier si nécessaire.

## 6. Utilisation

```bash
python main.py
```

L'exécution enchaîne tous les modules, produit les graphiques numérotés puis
exporte le classeur Excel avec tableau de bord.

**Paramètres principaux** (fonction `run_convex_optimization`) :

| Paramètre | Valeur par défaut | Signification |
|---|---|---|
| `w_max` | `0.15` | Plafond de pondération par banque |
| `lam_sys` | `1.0` | Pénalisation du risque systémique |
| `lam_fx` | `1.0` | Pénalisation du risque de change |

Le programme exige `N × w_max ≥ 1` : avec 35 banques, `w_max` doit être au
moins égal à environ 2,9 %.

## 7. Sorties

Les graphiques sont enregistrés au format PNG par `save_chart`, avec un
préfixe numérique. Ceux du module d'optimisation sont numérotés à partir de 48 :

| N° | Graphique |
|---|---|
| 48 | Comparaison des pondérations Min-Variance / ERC / bi-pénalisé |
| 49 | Contributions au risque de l'ERC |
| 50 | Frontière risque / exposition systémique |
| 51 | Frontière risque / exposition au change |
| 52 | Pondérations ERC vs Min-Variance bi-pénalisé |
| 53 | Comparaison des indices optimisés |
| 54 | Comparaison des indicateurs de concentration (HHI) |

Autres sorties : séries d'indices, tableaux statistiques, poids par méthode,
indicateurs (volatilité, HHI, nombre effectif de banques) et classeur Excel.

## 8. Principaux résultats

- Les portefeuilles Min-Variance et bi-pénalisé sont nettement plus concentrés
  que l'ERC : les deux premières banques atteignent le plafond de 15 %.
- L'ERC conserve un poids strictement positif sur toutes les banques
  (≈ 1,3 % à 7 %), car la formulation log-barrière interdit les poids nuls.
- La pénalisation exclut huit banques (nigérianes et sud-africaines) que la
  seule minimisation de variance laissait marginalement présentes.

> Valeurs lues sur les graphiques, à confirmer avec les sorties chiffrées.

## 9. Références principales

- Adrian, T., & Brunnermeier, M. K. (2016). CoVaR. *American Economic Review*.
- Girardi, G., & Ergün, A. T. (2013). Systemic risk measurement: multivariate
  GARCH estimation of CoVaR. *Journal of Banking & Finance*.
- Engle, R. (2002). Dynamic conditional correlation. *JBES*.
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing
  of financial correlation matrices. *Physical Review Letters*.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The properties of equally
  weighted risk contribution portfolios. *JPM*.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.

La bibliographie complète figure dans le rapport.

## 10. Auteure et encadrement

**Zineb Madani** — Étudiante en 2ᵉ année, filière Actuariat/Finance, INSEA.

Stage à Attijari Global Research (29 juin – 6 septembre 2026), encadré par
Abdelaziz Lahlou (Chief Economist), Walid Kabbaj et Soufiane Khelloufi.

Rapport : *Construction d'un indice bancaire panafricain — Ajustement au
risque systémique*.
