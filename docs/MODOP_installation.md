## 1. Création de l'environnement virtuel
Ouvrir PowerShell, se placer dans le dossier du projet :
```powershell
cd C:\CoreGuard-AI
python -m venv .venv
```

## 2. Activation de l'environnement virtuel
```powershell
.venv\Scripts\Activate.ps1
```
## 3. Installation des dépendances
```powershell
pip install -r requirements.txt
```
## 4. Lancement de la plateforme
```powershell
python -m streamlit run app.py
```
## 5. Installation sur VM
### 5.1 Transfert de l'archive vers la VM

Copier `CoreGuard-AI.zip` sur la VM via l'un des moyens suivants : partage réseau,
transfert de fichiers via le client Bureau à distance (RDP), ou téléchargement depuis
un lien cloud accessible depuis la VM. Décompresser ensuite à un emplacement simple,
par exemple `C:\CoreGuard-AI\`.

### 5.2 Accès à l'application depuis la VM

**à l'intérieur de la VM** et aller sur `http://localhost:8501`. Aucune configuration
réseau supplémentaire n'est nécessaire.

**Accès externe — depuis un navigateur en dehors de la VM :**
Si l'accès doit se faire depuis un poste externe sans passer par RDP, il faut :

1. Connaître l'adresse IP de la VM sur le réseau.
2. Ouvrir le port 8501 dans le pare-feu Windows de la VM (PowerShell en administrateur) :
   ```powershell
   New-NetFirewallRule -DisplayName "Streamlit" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
   ```
3. Accéder depuis le poste externe à `http://<IP_de_la_VM>:8501` — cette adresse est
   aussi affichée automatiquement dans le terminal au lancement de Streamlit, sous
   la ligne **« Network URL »**.

### 5.3 VM sans accès internet

Si la VM est isolée (pas d'accès à `pypi.org`), l'étape 3  (`pip install -r requirements.txt`)
échouera. Dans ce cas, télécharger au préalable les paquets sur une machine connectée
avec `pip download -r requirements.txt -d packages\` , transférer le dossier `packages\`
sur la VM, puis installer hors-ligne avec :
```powershell
pip install --no-index --find-links=packages -r requirements.txt
```

