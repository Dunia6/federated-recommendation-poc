import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Flatten, Concatenate, 
                                     Dense, Multiply, Dropout)
from tensorflow.keras.optimizers import Adam

def create_ncf_model(num_users, num_items, latent_dim_gmf=32, latent_dim_mlp=32,
                     mlp_layers_dims=[64, 32, 16, 8]):
    """
    Crée l'architecture complète du modèle Neural Collaborative Filtering (NCF).

    Args:
        num_users (int): Nombre total d'utilisateurs uniques.
        num_items (int): Nombre total d'articles (films) uniques.
        latent_dim_gmf (int): Dimension des vecteurs latents pour la branche GMF.
        latent_dim_mlp (int): Dimension des vecteurs latents pour la branche MLP.
        mlp_layers_dims (list): Liste d'entiers pour la taille des couches du MLP.

    Returns:
        tensorflow.keras.Model: Le modèle NCF non compilé.
    """

    # --- Entrées du modèle ---
    # On définit deux entrées : une pour l'ID de l'utilisateur, une pour l'ID de l'article.
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # --- Branche GMF (Generalized Matrix Factorization) ---
    # Couches d'embedding spécifiques à la branche GMF
    gmf_user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim_gmf, 
                                   name='gmf_user_embedding')(user_input)
    gmf_item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim_gmf, 
                                   name='gmf_item_embedding')(item_input)

    # Aplatir les vecteurs pour les opérations
    gmf_user_vec = Flatten(name='gmf_flatten_user')(gmf_user_embedding)
    gmf_item_vec = Flatten(name='gmf_flatten_item')(gmf_item_embedding)

    # Produit élément par élément (Hadamard product)
    gmf_vector = Multiply(name='gmf_product')([gmf_user_vec, gmf_item_vec])

    # --- Branche MLP (Multi-Layer Perceptron) ---
    # Couches d'embedding spécifiques à la branche MLP
    mlp_user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim_mlp, 
                                   name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim_mlp, 
                                   name='mlp_item_embedding')(item_input)

    # Aplatir les vecteurs
    mlp_user_vec = Flatten(name='mlp_flatten_user')(mlp_user_embedding)
    mlp_item_vec = Flatten(name='mlp_flatten_item')(mlp_item_embedding)

    # Concaténer les deux vecteurs pour former l'entrée du MLP
    mlp_input_vector = Concatenate(name='mlp_concat')([mlp_user_vec, mlp_item_vec])

    # Construire les couches denses du MLP de manière dynamique
    mlp_vector = mlp_input_vector
    for i, dim in enumerate(mlp_layers_dims):
        mlp_vector = Dense(dim, activation='relu', name=f'mlp_dense_{i}')(mlp_vector)
        

    # --- Fusion des deux branches (GMF et MLP) ---
    # Concaténer les vecteurs de sortie des deux branches
    final_vector = Concatenate(name='final_concat')([gmf_vector, mlp_vector])

    # --- Couche de prédiction finale ---
    # Une seule neurone avec une activation sigmoïde pour une sortie entre 0 et 1
    prediction = Dense(1, activation='sigmoid', name='prediction')(final_vector)

    # Création et retour du modèle Keras
    model = Model(inputs=[user_input, item_input], outputs=prediction, name='NCF_Model')

    return model


if __name__ == '__main__':
    # Paramètres d'exemple
    NUM_USERS_EXAMPLE = 6040  # Tiré de l'exploration des données MovieLens 1M
    NUM_ITEMS_EXAMPLE = 3952  # Tiré de l'exploration des données MovieLens 1M
    
    # Créer le modèle
    model = create_ncf_model(NUM_USERS_EXAMPLE, NUM_ITEMS_EXAMPLE)
    
    # Afficher le résumé de l'architecture
    print("Résumé de l'architecture du modèle NCF :")
    model.summary()
