# CoreGuard AI

Plateforme AIOps de détection d'anomalies et de classification d'incidents sur les KPI du réseau cœur, développée dans le cadre d'un stage chez Orange Tunisie.

## Fonctionnalités
- Détection d'anomalies sur les données réseau (PGW, SGW, PDC, Statistics, PM Job)
- Classification des incidents via XGBoost
- Analyse de causes racines (RCA)
- Génération de rapports PDF

## Structure du projet
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
## Installation
Voir [docs/MODOP_installation.md](docs/MODOP_installation.md) pour le guide d'installation complet.


