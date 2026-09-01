## CoreGuard AI

Plateforme AIOps de détection d'anomalies et de classification d'incidents sur les KPI du réseau cœur, développée dans le cadre d'un stage chez Orange Tunisie.

### Fonctionnalités
- Détection d'anomalies sur les données réseau (PGW, SGW, PDC, Statistics, PM Job)
- Classification des incidents via XGBoost
- Analyse de causes racines (RCA)
- Génération de rapports PDF

### Structure du projet
```
CoreGUARD/
├── app.py # Point d'entrée Streamlit
├── engine/ # Modules du pipeline ML
│ ├── features/ # Nettoyage et feature engineering
│ └── models/ # Entraînement et inférence
├── ui/ # Thème et composants d'interface
├── data/ # Données brutes, nettoyées, traitées
├── models/ # Modèles entraînés (.pkl)
├── docs/ # Documentation
└── outputs/ # Rapports générés
```
### Installation
Voir [docs/MODOP_installation.md](docs/MODOP_installation.md) pour le guide d'installation complet.

### Détection d'anomalies (Isolation Forest)
Modèle : Isolation Forest — détection non supervisée d'anomalies sur les KPI réseau.

- Dataset : 2777 observations, 194 variables
- Anomalies détectées : 278 (10.01%)
- Observations normales : 2499

#### Répartition des anomalies par heure

| Heure | Anomalies | Heure | Anomalies |
|-------|-----------|-------|-----------|
| 00h | 11 | 12h | 22 |
| 01h | 12 | 13h | 17 |
| 02h | 9  | 14h | 17 |
| 03h | 4  | 15h | 6  |
| 04h | 3  | 16h | 7  |
| 05h | 4  | 17h | 10 |
| 06h | 4  | 18h | 5  |
| 07h | 5  | 19h | 2  |
| 08h | 28 | 20h | 4  |
| 09h | 29 | 21h | 5  |
| 10h | 30 | 22h | 6  |
| 11h | 30 | 23h | 8  |

Un pic net d'anomalies apparaît entre 8h et 12h, correspondant probablement aux heures de forte charge réseau.

#### Répartition des anomalies par jour

| Jour | Anomalies |
|------|-----------|
| Lundi | 65 |
| Mardi | 65 |
| Mercredi | 44 |
| Jeudi | 40 |
| Vendredi | 29 |
| Samedi | 16 |
| Dimanche | 19 |

Les anomalies sont plus fréquentes en début de semaine (lundi-mardi) et diminuent le week-end.
### Performance du modèle

Modèle : XGBoost — classification binaire d'incidents, validé par cross-validation temporelle (5 folds).

**F1-score en cross-validation :** 0.751 ± 0.306

#### Seuil par défaut (0.500)

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0 (normal) | 0.910 | 1.000 | 0.953 | 487 |
| 1 (incident) | 1.000 | 0.304 | 0.467 | 69 |

- Accuracy : 0.914
- ROC-AUC : 0.995

#### Seuil optimal (0.034, maximisant le F1)

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0 (normal) | 0.992 | 0.982 | 0.987 | 487 |
| 1 (incident) | 0.878 | 0.942 | 0.909 | 69 |

- Accuracy : 0.977
- ROC-AUC : 0.995
- F1-score : 0.909

Le seuil optimal a été retenu pour maximiser la détection des incidents (recall) tout en gardant une bonne précision, plutôt que le seuil par défaut de 0.5.

#### Top features (importance XGBoost)

| Feature | Importance |
|---------|------------|
| `pdc__num-cmds-aapn-ok__diff_15min` | 0.082 |
| `pgw__eps-bearer-creation-attempted__delta` | 0.075 |
| `stats__dt_error_rate__diff_15min` | 0.060 |
| `pmjob__memory_usage_rate__roll_std_1h` | 0.057 |
| `pgw__eps-bearer-creation-attempted__delta__rolling` | 0.056 |
