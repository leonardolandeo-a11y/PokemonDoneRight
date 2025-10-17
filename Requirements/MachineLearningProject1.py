#Imported libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd   
from sklearn.cluster import KMeans
#Read csv
Pokemon = pd.read_csv("smogon.csv")

#Necessary Columns
MovimientosPokemon = Pokemon["moves"].copy()
NombresPokemon = Pokemon["Pokemon"].copy()

#Instance of TF-IDF
TextToNumbers = TfidfVectorizer(ngram_range=(1,1)) #We're using unigram Ex: ("Hola","como","estas")

#Matrix of Numeric values
MatrixPokemonNumericValues = TextToNumbers.fit_transform(MovimientosPokemon)

print(f"TF-IDF:\n{MatrixPokemonNumericValues.toarray()}")
print("\n")
#Instance of KMeans
Cluster = KMeans(n_clusters=18) #We're using 18 clusters. Because we have 18 classes of pokemon 
MatrixOfClusters = Cluster.fit(MatrixPokemonNumericValues)
MatrixClusterPokemon = MatrixOfClusters.labels_
print(f"Clusters:\n{MatrixClusterPokemon}")

print("\n")

#Tokens (Elements of the vocabulary)
Vocabulary = TextToNumbers.vocabulary_
print(f"Tokens:\n{Vocabulary}")
print("\n")

#keys of Tokens
Names = TextToNumbers.get_feature_names_out()
print(f"Name:\n{Names}")

#Final DataFrame
PokemonDataSet = pd.DataFrame(MatrixPokemonNumericValues.toarray(), columns = Names)
PokemonDataSet["ClusterPokemon"] = MatrixClusterPokemon
print(f"PokemonDataSet:\n{PokemonDataSet}")

#Final CSV
SerieClustersPokemon = pd.Series(MatrixClusterPokemon,name = "ClusterPokemon")

PokemonDataSet_Name_Cluster = pd.concat([NombresPokemon, SerieClustersPokemon],axis= 1)

PokemonDataSet_Name_Cluster.to_csv("PokemonDataset.csv")