# Documentation Technique : « Mise en oeuvre d’un modèle d'apprentissage profond fédéré pour un système de recommandation personnalisé »

---

**Projet Tuteuré - Master 1 Data Science**  
**Université  Don Bosco de Lubumbashi**  
**Année Académique :** 2024-2025

**Étudiants :**
- MASIMANGO DUNIA JEPHTE
- JOSEPH KABEYA

**Encadrante :**  
- Dr-Ing. Olfa FERCHICHI

---

## Table des Matières

1. [Vue d'Ensemble du Projet](#1-vue-densemble-du-projet)
2. [Architecture du Système](#2-architecture-du-système)
3. [Installation et Configuration](#3-installation-et-configuration)
4. [Structure du Code](#4-structure-du-code)
5. [Guide d'Utilisation](#5-guide-dutilisation)
6. [Analyse Technique](#6-analyse-technique)
7. [Résultats et Évaluation](#7-résultats-et-évaluation)
8. [Troubleshooting](#8-troubleshooting)
9. [Références](#9-références)

---

## 1. Vue d'Ensemble du Projet

### 1.1 Objectif

Ce projet implémente et compare deux approches d'entraînement pour un système de recommandation basé sur Neural Collaborative Filtering (NCF) :
- **Approche Centralisée** : Entraînement classique sur un serveur central
- **Approche Fédérée** : Entraînement distribué avec TensorFlow Federated (TFF)

### 1.2 Dataset

**MovieLens 1M** : 
- 6,040 utilisateurs
- 3,706 films  
- ~1 million d'interactions
- Format : ratings.dat, users.dat, movies.dat

### 1.3 Métrique d'Évaluation Principale

**AUC (Area Under ROC Curve)** : Mesure la capacité du modèle à distinguer les interactions positives des négatives dans un contexte de classification binaire.

---

## 2. Architecture du Système

### 2.1 Architecture NCF (Neural Collaborative Filtering)

Le modèle combine deux approches complémentaires :

```
Utilisateur ID  ──┐                    ┌── Film ID
                  │                    │
                  ▼                    ▼
            GMF Embedding        MLP Embedding
                  │                    │
                  ▼                    ▼
            Produit Hadamard     Concaténation
                  │                    │
                  ▼                    ▼
              GMF Vector          Couches Denses
                  │                    │
                  └──── Fusion ────────┘
                          │
                          ▼
                   Couche Prédiction
                          │
                          ▼
                  Probabilité [0,1]
```

#### Composants :
- **GMF (Generalized Matrix Factorization)** : Capture les interactions linéaires
- **MLP (Multi-Layer Perceptron)** : Capture les interactions non-linéaires
- **Fusion** : Combine les deux approches pour la prédiction finale

### 2.2 Architecture Fédérée

```
                    Serveur Central
                    (Agrégation)
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       Client 1     Client 2     Client N
     (Utilisateur) (Utilisateur) (Utilisateur)
         │            │            │
         ▼            ▼            ▼
    Données Locales ──────────────────
    (Interactions Privées)
```

#### Processus FedAvg :
1. **Initialisation** : Modèle global partagé
2. **Sélection** : Échantillonnage de clients actifs
3. **Entraînement Local** : Chaque client entraîne sur ses données
4. **Agrégation** : Moyennage pondéré des modèles clients
5. **Mise à jour** : Nouveau modèle global distribué

---

## 3. Installation et Configuration

### 3.1 Prérequis Système

```bash
# Système d'exploitation
Windows 10/11 ou Linux/WSL

# Python
Python 3.8+ (recommandé : 3.10)

# Mémoire
RAM : 8GB minimum, 16GB recommandé

# Stockage
Espace libre : 5GB minimum
```

### 3.2 Installation des Dépendances

```bash
# Cloner le projet
git clone [URL_DU_PROJET]
cd federated-recommendation-poc

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Windows
.venv\\Scripts\\activate
# Linux/WSL
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3.3 Configuration des Données

```bash
# Créer le dossier de données
mkdir -p data/ml-1m

# Télécharger MovieLens 1M depuis
# https://grouplens.org/datasets/movielens/1m/

# Extraire dans data/ml-1m/
# Structure attendue :
# data/ml-1m/
#   ├── ratings.dat
#   ├── users.dat
#   └── movies.dat
```

### 3.4 Vérification de l'Installation

```bash
# Test du chargement des données
python src/data_pipeline/loader.py

# Test de l'architecture du modèle
python src/models/ncf_model.py
```

---

## 4. Structure du Code

### 4.1 Organisation des Dossiers

```
federated-recommendation-poc/
├── data/                          # Données brutes
│   └── ml-1m/                    # Dataset MovieLens 1M
├── src/                          # Code source principal
│   ├── __init__.py
│   ├── config.py                 # Configuration globale
│   ├── train_centralized.py     # Script d'entraînement centralisé
│   ├── train_federated.py       # Script d'entraînement fédéré
│   ├── train_federated_manual.py # Implémentation FedAvg manuelle
│   ├── data_pipeline/           # Pipeline de données
│   │   ├── __init__.py
│   │   ├── loader.py            # Chargement des données
│   │   └── federated_loader.py  # Préparation données fédérées
│   └── models/                  # Architectures de modèles
│       ├── __init__.py
│       └── ncf_model.py         # Modèle NCF
├── notebooks/                   # Analyses et visualisations
│   ├── 01-Data_Exploration.ipynb
│   └── 02-Results_Analysis.ipynb
├── results/                     # Résultats d'entraînement
├── requirements.txt             # Dépendances Python
├── EXPERIMENTAL_PROTOCOL.md     # Protocole expérimental
└── README.md                    # Documentation utilisateur
```

### 4.2 Modules Principaux

#### 4.2.1 Configuration (`src/config.py`)

```python
# Hyperparamètres du modèle
LATENT_DIM_GMF = 32              # Dimension embedding GMF
LATENT_DIM_MLP = 32              # Dimension embedding MLP
MLP_LAYERS = [64, 32, 16, 8]     # Architecture MLP

# Paramètres d'entraînement
BATCH_SIZE = 256                 # Taille des lots
LEARNING_RATE = 0.001            # Taux d'apprentissage
NUM_NEGATIVE_SAMPLES = 4         # Ratio échantillonnage négatif

# Configuration fédérée
NUM_CLIENTS = 100                # Nombre de clients simulés
NUM_ROUNDS = 50                  # Rounds de communication
CLIENTS_PER_ROUND = 10           # Clients participants par round
LOCAL_EPOCHS = 1                 # Époques d'entraînement local
```

#### 4.2.2 Modèle NCF (`src/models/ncf_model.py`)

**Fonction principale :** `create_ncf_model(num_users, num_items, ...)`

**Paramètres :**
- `num_users` : Nombre total d'utilisateurs
- `num_items` : Nombre total d'items
- `latent_dim_gmf` : Dimension latente pour GMF
- `latent_dim_mlp` : Dimension latente pour MLP
- `mlp_layers_dims` : Liste des tailles des couches MLP

**Retour :** Modèle Keras non compilé

#### 4.2.3 Chargement des Données (`src/data_pipeline/`)

##### `loader.py`
- **Fonction :** `load_movielens_1m(data_path)`
- **Rôle :** Charge les fichiers .dat et retourne les DataFrames pandas
- **Retour :** Tuple (ratings_df, users_df, movies_df)

##### `federated_loader.py`
- **Fonction :** `create_federated_datasets(data_dir, num_clients, ...)`
- **Rôle :** Prépare les datasets TensorFlow pour simulation fédérée
- **Retour :** Liste de tf.data.Dataset par client + mappings

---

## 5. Guide d'Utilisation

### 5.1 Workflow Standard

```bash
# 1. Exploration des données
jupyter notebook notebooks/01-Data_Exploration.ipynb

# 2. Entraînement de la baseline centralisée
python src/train_centralized.py

# 3. Entraînement fédéré
python src/train_federated.py

# 4. Analyse des résultats
jupyter notebook notebooks/02-Results_Analysis.ipynb
```

### 5.2 Entraînement Centralisé

**Script :** `src/train_centralized.py`

**Processus :**
1. Chargement et mapping des données
2. Negative sampling (4:1 ratio)
3. Split train/validation (80/20)
4. Création du modèle NCF
5. Entraînement avec Adam optimizer
6. Sauvegarde des métriques dans `results/centralized_results.csv`

**Configuration :**
```python
CENTRALIZED_EPOCHS = 10      # Nombre d'époques
BATCH_SIZE = 256            # Taille des lots
LEARNING_RATE = 0.001       # Taux d'apprentissage
```

### 5.3 Entraînement Fédéré

**Script :** `src/train_federated_manual.py`

**Processus :**
1. Préparation des datasets clients
2. Définition du modèle TFF
3. Configuration du processus FedAvg
4. Simulation des rounds de communication
5. Sauvegarde des métriques dans `results/federated_results_*.csv`

**Configuration :**
```python
NUM_CLIENTS = 100           # Clients totaux
CLIENTS_PER_ROUND = 10      # Clients actifs par round
NUM_ROUNDS = 50             # Rounds de communication
LOCAL_EPOCHS = 1            # Époques locales
```

### 5.4 Personnalisation des Hyperparamètres

**Modification dans `config.py` :**

```python
# Pour améliorer les performances fédérées
LOCAL_EPOCHS = 3            # Augmenter l'entraînement local
CLIENTS_PER_ROUND = 20      # Plus de participation
LEARNING_RATE = 0.003       # Ajuster le learning rate

# Pour des expériences rapides
NUM_CLIENTS = 50            # Moins de clients
NUM_ROUNDS = 25             # Moins de rounds
```

---

## 6. Analyse Technique

### 6.1 Pipeline de Données

#### Preprocessing Steps

1. **Chargement** : Lecture des fichiers .dat avec pandas
2. **Mapping** : Conversion des IDs vers des indices continus
3. **Negative Sampling** : Génération d'interactions négatives (ratio 4:1)
4. **Partitioning** : Distribution des données par utilisateur (fédéré)
5. **Batching** : Création des lots TensorFlow

#### Stratégie de Negative Sampling

```python
# Pour chaque utilisateur
positive_items = user_interactions
negative_items = all_items - positive_items
sampled_negatives = random_sample(negative_items, 
                                 len(positive_items) * 4)
```

### 6.2 Architecture du Modèle

#### Dimensions des Embeddings

```python
# Configuration par défaut
User Embedding (GMF): [num_users, 32]
Item Embedding (GMF): [num_items, 32]
User Embedding (MLP): [num_users, 32]  
Item Embedding (MLP): [num_items, 32]

# Couches MLP
Dense Layer 1: [64, relu]
Dense Layer 2: [32, relu]
Dense Layer 3: [16, relu]
Dense Layer 4: [8, relu]

# Fusion et prédiction
Concatenation: [GMF_output + MLP_output]
Final Dense: [1, sigmoid]
```

#### Fonction de Perte

**Binary Crossentropy** : Appropriée pour la classification binaire (interaction/non-interaction)

### 6.3 Processus Fédéré

#### Algorithme FedAvg

```python
# Pseudocode
for round_t in range(NUM_ROUNDS):
    # 1. Sélection des clients
    selected_clients = random_sample(all_clients, CLIENTS_PER_ROUND)
    
    # 2. Entraînement local
    local_models = []
    for client in selected_clients:
        local_model = copy(global_model)
        local_model.train(client_data, LOCAL_EPOCHS)
        local_models.append(local_model)
    
    # 3. Agrégation
    global_model = weighted_average(local_models)
    
    # 4. Évaluation
    metrics = evaluate(global_model, validation_data)
```

---

## 7. Résultats et Évaluation

### 7.1 Métriques d'Évaluation

#### Métriques Principales
- **AUC (Area Under ROC Curve)** : Métrique principale pour l'évaluation
- **Accuracy** : Pourcentage de prédictions correctes
- **Loss** : Binary Crossentropy

#### Interprétation des Résultats

```python
# AUC Interpretation
AUC >= 0.9  : Excellent
AUC >= 0.8  : Très bon
AUC >= 0.7  : Bon
AUC >= 0.6  : Acceptable
AUC <  0.6  : Problématique
AUC ~  0.5  : Performance aléatoire
```

### 7.2 Résultats Typiques

#### Performance Centralisée (Baseline)
```
Époque 10:
├── AUC de validation: 0.92
├── Accuracy: 0.88
└── Loss: 0.29
```

#### Performance Fédérée (Problématique)
```
Round 50:
├── AUC moyenne: 0.49 (❌ Niveau aléatoire)
├── Accuracy: 0.80 (⚠️ Bloquée)
└── Loss: 0.68 (❌ Élevée)
```

### 7.3 Analyse des Problèmes Identifiés

#### Problèmes Critiques
1. **AUC au niveau du hasard** (~0.5)
2. **Accuracy figée** à 80%
3. **Absence de convergence** observable
4. **Hyperparamètres inadaptés**

#### Solutions Recommandées
1. **Réduire le learning rate** : 0.01 → 0.003
2. **Augmenter les époques locales** : 1 → 3
3. **Améliorer la participation** : 10% → 20%
4. **Vérifier l'implémentation FedAvg**

---

## 8. Troubleshooting

### 8.1 Problèmes Courants

#### Erreur : "Module 'config' introuvable"
```bash
# Solution
cd federated-recommendation-poc  # S'assurer d'être à la racine
python -m src.train_centralized  # Utiliser le module path
```

#### Erreur : "Données introuvables"
```bash
# Vérifier la structure
ls data/ml-1m/
# Doit contenir: ratings.dat, users.dat, movies.dat

# Solution pour WSL
# Utiliser des chemins absolus dans config.py
```

#### Erreur TensorFlow Federated
```bash
# Vérifier la compatibilité des versions
pip list | grep tensorflow
# TensorFlow et TFF doivent être compatibles
```

#### Performance fédérée dégradée
```python
# Dans config.py, ajuster:
LOCAL_EPOCHS = 3        # Plus d'entraînement local
LEARNING_RATE = 0.003   # Learning rate plus conservateur
CLIENTS_PER_ROUND = 20  # Plus de participation
```

### 8.2 Debug et Monitoring

#### Logs de Débogage
```python
# Ajouter dans les scripts
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier les shapes des tenseurs
print(f"Input shape: {input_tensor.shape}")
print(f"Model output shape: {output.shape}")
```

#### Monitoring des Métriques
```python
# Visualisation en temps réel
import matplotlib.pyplot as plt

def plot_training_progress(metrics):
    plt.plot(metrics['loss'])
    plt.title('Training Loss')
    plt.show()
```

---

## 9. Références

### 9.1 Bibliographie Technique

1. **Neural Collaborative Filtering**
   - He, X., et al. (2017). "Neural Collaborative Filtering"
   - Paper: https://arxiv.org/abs/1708.05031

2. **Federated Learning**
   - McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
   - Paper: https://arxiv.org/abs/1602.05629

3. **TensorFlow Federated**
   - Documentation: https://www.tensorflow.org/federated
   - API Reference: https://www.tensorflow.org/federated/api_docs

### 9.2 Datasets

- **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
- **Documentation**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets

### 9.3 Outils et Frameworks

- **TensorFlow**: 2.x
- **TensorFlow Federated**: Compatible version
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization

---

## Annexes

### A. Commandes Utiles

```bash
# Installation rapide
pip install tensorflow tensorflow-federated pandas numpy matplotlib seaborn jupyter

# Test de l'environnement
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow_federated as tff; print(tff.__version__)"

# Nettoyage des résultats
rm -rf results/*.csv

# Execution complète
python src/train_centralized.py && python src/train_federated.py
```

### B. Configuration Recommandée pour Production

```python
# config.py - Configuration optimisée
LATENT_DIM_GMF = 64
LATENT_DIM_MLP = 64
MLP_LAYERS = [128, 64, 32, 16]
LOCAL_EPOCHS = 5
CLIENTS_PER_ROUND = 20
LEARNING_RATE = 0.001
```

### C. Métriques de Performance Système

```bash
# Monitoring des ressources
htop                    # CPU/Memory usage
nvidia-smi             # GPU usage (si disponible)
du -sh .venv/          # Espace disque des dépendances
```

---

*Documentation générée automatiquement*  
*Dernière mise à jour: 24 juillet 2025*
