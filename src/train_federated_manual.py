import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Importer nos modules et la nouvelle configuration
from data_pipeline.federated_loader import create_federated_datasets
from models.ncf_model import create_ncf_model
import config

# --- Préparation des données fédérées ---
print("="*50)
print("Démarrage de l'entraînement Fédéré Manuel")
print("="*50)

# Note : on utilise NUM_CLIENTS_TO_SIMULATE du fichier config
federated_train_data, user_map, movie_map = create_federated_datasets(
    config.DATA_DIR, config.NUM_CLIENTS_TO_SIMULATE,
    num_negative_samples=config.NUM_NEGATIVE_SAMPLES,
    batch_size=config.BATCH_SIZE
)

num_users = len(user_map)
num_items = len(movie_map)

# --- Création du Modèle Global ---
global_model = create_ncf_model(
    num_users, num_items,
    latent_dim_gmf=config.LATENT_DIM_GMF,
    latent_dim_mlp=config.LATENT_DIM_MLP,
    mlp_layers_dims=config.MLP_LAYERS
)

# Fonction utilitaire pour la moyenne des poids
def average_weights(weight_list):
    avg_weights = []
    for weights_in_layer in zip(*weight_list):
        avg_weights.append(np.mean(np.array(weights_in_layer), axis=0))
    return avg_weights

# --- Boucle FedAvg Manuelle et Logging ---
print("\n" + "="*50)
print(f"Entraînement pour {config.NUM_ROUNDS} rounds...")
print("="*50)

# Préparer le fichier de log
results_history = []

for rnd in range(1, config.NUM_ROUNDS + 1):
    local_weights = []
    local_metrics = []
    
    # Échantillonner les clients
    available_clients = len(federated_train_data)
    sampled_idxs = np.random.choice(
        available_clients, size=min(config.CLIENTS_PER_ROUND, available_clients), replace=False
    )
    
    # Entraînement local
    for client_id in tqdm(sampled_idxs, desc=f"Round {rnd}/{config.NUM_ROUNDS}"):
        local_model = create_ncf_model(num_users, num_items) # Créer un nouveau modèle pour la compilation
        local_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        # Définir les poids du modèle local comme étant ceux du modèle global
        local_model.set_weights(global_model.get_weights())
        
        # Entraînement local
        history = local_model.fit(federated_train_data[client_id], epochs=config.LOCAL_EPOCHS, verbose=0)
        
        local_weights.append(local_model.get_weights())
        # Sauvegarder la dernière valeur des métriques pour ce client
        local_metrics.append({k: v[-1] for k, v in history.history.items()})

    # Agrégation des poids
    new_global_weights = average_weights(local_weights)
    global_model.set_weights(new_global_weights)

    # Calcul des métriques moyennes du round
    avg_metrics = pd.DataFrame(local_metrics).mean().to_dict()
    avg_metrics['round'] = rnd
    
    print(f"Round {rnd:2d} — Metrics: { {k: round(v, 4) for k, v in avg_metrics.items()} }")
    results_history.append(avg_metrics)

# --- Sauvegarde des résultats ---
results_df = pd.DataFrame(results_history)
results_df.to_csv(config.FEDERATED_RESULTS_FILE, index=False)

print("\n--- Simulation Fédérée Terminée ---")
print(f"Les résultats de l'entraînement ont été sauvegardés dans '{config.FEDERATED_RESULTS_FILE}'")