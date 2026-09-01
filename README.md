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
