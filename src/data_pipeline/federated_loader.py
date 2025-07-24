import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .loader import load_movielens_1m

def create_federated_datasets(data_dir, num_clients, num_negative_samples=4, batch_size=32):
    """
    Charge, prétraite et partitionne les données MovieLens 1M pour un scénario fédéré.

    Args:
        data_dir (str): Chemin vers le dossier ml-1m.
        num_clients (int): Le nombre total de clients à simuler.
        num_negative_samples (int): Nombre d'échantillons négatifs par positif.
        batch_size (int): Taille des lots pour les datasets clients.

    Returns:
        list: Une liste de tf.data.Dataset, un pour chaque client.
        dict: Le mapping des utilisateurs.
        dict: Le mapping des films.
    """
    print("--- Démarrage de la préparation des données fédérées ---")
    
    # 1. Chargement et Mapping des IDs (similaire à la version centralisée)
    ratings_df_raw, _, _ = load_movielens_1m(data_dir)

    user_ids = ratings_df_raw['UserID'].unique().tolist()
    movie_ids = ratings_df_raw['MovieID'].unique().tolist()
    
    # On s'assure de ne pas avoir plus de clients que d'utilisateurs réels
    if num_clients > len(user_ids):
        raise ValueError("Le nombre de clients ne peut pas excéder le nombre d'utilisateurs uniques.")

    # On ne prend que les `num_clients` premiers utilisateurs pour la simulation
    selected_user_ids = user_ids[:num_clients]
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(selected_user_ids)}
    movie_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
    
    df = ratings_df_raw[ratings_df_raw['UserID'].isin(selected_user_ids)].copy()
    df['UserID'] = df['UserID'].map(user_map)
    df['MovieID'] = df['MovieID'].map(movie_map)

    print(f"Données mappées pour {num_clients} clients.")

    # 2. Negative Sampling (similaire à la version centralisée)
    train_df = df[['UserID', 'MovieID']]
    train_df['label'] = 1.0 # Utiliser des flottants pour la perte

    all_mapped_movie_ids = list(movie_map.values())
    negative_samples = []

    for user_id in tqdm(range(num_clients), desc="Génération d'échantillons négatifs"):
        positive_movie_ids = train_df[train_df['UserID'] == user_id]['MovieID'].values
        negative_movie_ids = np.setdiff1d(all_mapped_movie_ids, positive_movie_ids)
        num_to_sample = len(positive_movie_ids) * num_negative_samples
        num_samples = min(num_to_sample, len(negative_movie_ids))

        if num_samples > 0:
            samples = np.random.choice(negative_movie_ids, size=num_samples, replace=False)
            for movie_id in samples:
                negative_samples.append([user_id, movie_id, 0.0])

    negative_df = pd.DataFrame(negative_samples, columns=['UserID', 'MovieID', 'label'])
    full_df = pd.concat([train_df, negative_df]).sample(frac=1, random_state=42)

    # 3. Création des tf.data.Dataset pour chaque client
    client_datasets = []
    print(f"\nCréation de {num_clients} datasets clients...")
    for client_id in tqdm(range(num_clients), desc="Création des datasets clients"):
        client_df = full_df[full_df['UserID'] == client_id]
        
        if client_df.empty:
            continue # Ignorer les clients sans données

        # TFF attend les données sous forme de dictionnaire ordonné (OrderedDict)
        # Mais un simple dictionnaire fonctionne aussi pour `from_tensor_slices`
        features = {
            'user_input': np.array(client_df['UserID'], dtype=np.int32),
            'item_input': np.array(client_df['MovieID'], dtype=np.int32)
        }
        labels = np.array(client_df['label'], dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=len(client_df)).batch(batch_size)
        
        client_datasets.append(dataset)

    print("--- Préparation des données fédérées terminée. ---")
    return client_datasets, user_map, movie_map