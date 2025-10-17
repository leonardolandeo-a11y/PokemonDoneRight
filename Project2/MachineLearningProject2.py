#Imported Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans

#Read CSV
Pokemon = pd.read_csv("Project2\smogon.csv")

#List of controled vocabulary
ControledVocabulary =  [
    'bug', 'dark', 'dragon', 'electric', 'fairy', 
    'fighting', 'fire', 'flying', 'ghost', 'grass',
    'ground', 'ice', 'poison', 'psychic', 'rock', 
    'steel','water'
]

#Clean the data
for TypePokemon in ControledVocabulary:
    Pokemon["moves"] = Pokemon["moves"].str.replace(TypePokemon, f" {TypePokemon} ", regex=False)

Pokemon["moves"] = Pokemon["moves"].str.replace(r"\s+", " ", regex=True).str.strip()
PokemonMoves = Pokemon["moves"]

#Instances of TF-IDF
TextToNumbersOfVocabulary = TfidfVectorizer(vocabulary=ControledVocabulary,ngram_range=(1,1))

#Executing TF-IDF
MatrixOf_TextToNumbersOfVocabulary = TextToNumbersOfVocabulary.fit_transform(PokemonMoves)
print(f"Matrix of TF-IDF:\n{MatrixOf_TextToNumbersOfVocabulary.toarray()}")

#Instance of KMeans
Clusters = KMeans(n_clusters=17)
ClustersOfPokemonRespectVocabulary = Clusters.fit(MatrixOf_TextToNumbersOfVocabulary).labels_

print(f"Clusters of each Pokemon:\n{ClustersOfPokemonRespectVocabulary}")

#Final Dataframe
ClustersOfPokemonRespectVocabularySerie = pd.Series(ClustersOfPokemonRespectVocabulary,name="ClusterPokemon")

PokemonDataSetClusterVocabulary = pd.concat([Pokemon["Pokemon"],ClustersOfPokemonRespectVocabularySerie],axis = 1)
PokemonDataSetClusterVocabulary.to_csv("PokemonDataSetClusterVocabulary.csv")