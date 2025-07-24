import pandas as pd
import os

def load_movielens_1m(data_path):
    """
    Charge et retourne les trois dataframes du jeu de données MovieLens 1M.

    Args:
        data_path (str): Le chemin vers le dossier 'ml-1m'.

    Returns:
        tuple: Un tuple contenant les dataframes (ratings, users, movies).
    """
    # Définir les chemins vers les fichiers .dat
    ratings_path = os.path.join(data_path, 'ratings.dat')
    users_path = os.path.join(data_path, 'users.dat')
    movies_path = os.path.join(data_path, 'movies.dat')

    # Charger le fichier ratings.dat
    ratings = pd.read_csv(ratings_path, sep='::', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                          engine='python', encoding='latin-1')

    # Charger le fichier users.dat
    users = pd.read_csv(users_path, sep='::', header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                        engine='python', encoding='latin-1')

    # Charger le fichier movies.dat
    movies = pd.read_csv(movies_path, sep='::', header=None,
                         names=['MovieID', 'Title', 'Genres'],
                         engine='python', encoding='latin-1')

    print("Données chargées avec succès.")
    print(f"Nombre de notes : {len(ratings)}")
    print(f"Nombre d'utilisateurs : {len(users)}")
    print(f"Nombre de films : {len(movies)}")

    return ratings, users, movies

# Cette partie permet de tester le script directement
if __name__ == '__main__':
    # Chemin relatif depuis la racine du projet
    path = os.path.join('data', 'ml-1m')
    
    # Vérifier si les données existent
    if not os.path.exists(path):
        print(f"Le dossier de données '{path}' est introuvable.")
        print("Veuillez télécharger et décompresser le jeu de données MovieLens 1M.")
    else:
        # Appeler la fonction pour charger les données
        ratings_df, users_df, movies_df = load_movielens_1m(path)
        
        print("\nExtrait des notes :")
        print(ratings_df.head())
        
        print("\nExtrait des utilisateurs :")
        print(users_df.head())
        
        print("\nExtrait des films :")
        print(movies_df.head())