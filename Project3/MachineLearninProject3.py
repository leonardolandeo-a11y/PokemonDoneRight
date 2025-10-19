#Imported Libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

#Read CSV
Pokemon = pd.read_csv("Porject3/smogon.csv")

#Controlled Vocabulary list
ControlledVocabulary =  [      #17 types of Pokemons
    'bug', 'dark', 'dragon', 'electric', 'fairy', 
    'fighting', 'fire', 'flying', 'ghost', 'grass',
    'ground', 'ice', 'poison', 'psychic', 'rock', 
    'steel','water'
]

#Cleaning Data
for TypePokemon in ControlledVocabulary:
    Pokemon["moves"] = Pokemon["moves"].str.replace(TypePokemon, f" {TypePokemon} ",regex= False) 
Pokemon["moves"] = Pokemon["moves"].str.replace(r"\s+", " ", regex=True).str.strip()
#Example: TheTypeOfThePokemonIsFire -> The Type Of The Pokemon Is Fire
PokemonMoves = Pokemon["moves"]

#Instance of CountVectorizer
Count = CountVectorizer(vocabulary=ControlledVocabulary)

#Matrix of the number of repetitions
MatrixNumberRepetitions = Count.fit_transform(PokemonMoves).toarray()
print(MatrixNumberRepetitions)

#DataFrame of Pokemon (Without Dominant Type)
PokemonDataSet = pd.DataFrame(MatrixNumberRepetitions,columns=ControlledVocabulary)
PokemonDataSet.insert(0,"PokemonNames",Pokemon["Pokemon"])

#Final DataFrame

MostFrequentlyAbility = PokemonDataSet[ControlledVocabulary].idxmax(axis = 1)
PokemonDataSet.insert(1,"DominantType",MostFrequentlyAbility)
print(PokemonDataSet)