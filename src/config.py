import os

# --- Configuration du Jeu de Données et du Modèle ---
DATA_DIR = os.path.join('data', 'ml-1m')
NUM_NEGATIVE_SAMPLES = 4  # Nombre d'échantillons négatifs par interaction positive

# Hyperparamètres du modèle NCF
LATENT_DIM_GMF = 32
LATENT_DIM_MLP = 32
MLP_LAYERS = [64, 32, 16, 8]

# --- Configuration de l'Entraînement ---
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# --- Paramètres Spécifiques à l'Entraînement Centralisé ---
CENTRALIZED_EPOCHS = 10 # Nombre d'époques pour la baseline

# --- Paramètres Spécifiques à l'Entraînement Fédéré ---
# IMPORTANT : Le nombre de clients DOIT correspondre au nombre d'utilisateurs uniques
# Nous le déterminerons dynamiquement, mais on peut fixer une limite pour les tests
NUM_CLIENTS_TO_SIMULATE = 100 # Ex: simuler sur les 100 premiers utilisateurs
NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 1 # Nombre d'époques d'entraînement sur le client à chaque round

# --- Configuration du Logging ---
# Créer un dossier pour sauvegarder les résultats
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Noms des fichiers de résultats
CENTRALIZED_RESULTS_FILE = os.path.join(RESULTS_DIR, 'centralized_results.csv')
FEDERATED_RESULTS_FILE = os.path.join(RESULTS_DIR, f'federated_results_{NUM_CLIENTS_TO_SIMULATE}c_{CLIENTS_PER_ROUND}k_{NUM_ROUNDS}r.csv')