import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam

# Importer nos modules et la nouvelle configuration
from data_pipeline.loader import load_movielens_1m
from models.ncf_model import create_ncf_model
import config

# --- Chargement et Préparation des Données ---
print("="*50)
print("Démarrage de l'entraînement de la Baseline Centralisée")
print("="*50)

ratings_df_raw, _, _ = load_movielens_1m(config.DATA_DIR)

# Mapping des IDs
user_ids = ratings_df_raw['UserID'].unique().tolist()
movie_ids = ratings_df_raw['MovieID'].unique().tolist()
user_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
movie_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}

df = ratings_df_raw.copy()
df['UserID'] = df['UserID'].map(user_map)
df['MovieID'] = df['MovieID'].map(movie_map)

num_users = len(user_map)
num_items = len(movie_map)

# Préparation du jeu de données (Negative Sampling)
train_df = df[['UserID', 'MovieID']]
train_df['label'] = 1.0

all_mapped_movie_ids = list(movie_map.values())
negative_samples = []

for user_id in tqdm(train_df['UserID'].unique(), desc="Génération d'échantillons négatifs"):
    # ... (logique de negative sampling identique à avant) ...
    positive_movie_ids = train_df[train_df['UserID'] == user_id]['MovieID'].values
    negative_movie_ids = np.setdiff1d(all_mapped_movie_ids, positive_movie_ids)
    num_to_sample = len(positive_movie_ids) * config.NUM_NEGATIVE_SAMPLES
    num_samples = min(num_to_sample, len(negative_movie_ids))
    if num_samples > 0:
        samples = np.random.choice(negative_movie_ids, size=num_samples, replace=False)
        for movie_id in samples:
            negative_samples.append([user_id, movie_id, 0.0])

negative_df = pd.DataFrame(negative_samples, columns=['UserID', 'MovieID', 'label'])
full_df = pd.concat([train_df, negative_df]).sample(frac=1, random_state=42)

X = full_df[['UserID', 'MovieID']]
y = full_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Création et Compilation du Modèle ---
model = create_ncf_model(
    num_users, num_items,
    latent_dim_gmf=config.LATENT_DIM_GMF,
    latent_dim_mlp=config.LATENT_DIM_MLP,
    mlp_layers_dims=config.MLP_LAYERS
)
model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# --- Entraînement et Logging ---
print("\n" + "="*50)
print(f"Entraînement pour {config.CENTRALIZED_EPOCHS} époques...")
print("="*50)

# Callback pour sauvegarder les logs à la fin de chaque époque
log_callback = tf.keras.callbacks.CSVLogger(config.CENTRALIZED_RESULTS_FILE, separator=',', append=False)

history = model.fit(
    [X_train['UserID'], X_train['MovieID']],
    y_train,
    batch_size=config.BATCH_SIZE,
    epochs=config.CENTRALIZED_EPOCHS,
    verbose=1,
    validation_data=([X_test['UserID'], X_test['MovieID']], y_test), # Évaluer sur le test set à chaque époque
    callbacks=[log_callback]
)

print("\nEntraînement terminé.")
print(f"Les résultats de l'entraînement ont été sauvegardés dans '{config.CENTRALIZED_RESULTS_FILE}'")