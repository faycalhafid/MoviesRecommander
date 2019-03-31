import pandas as pd
import numpy as np
"""
----------------------------------------------------------------------------
Partie 0 : -> Importer les données 
           -> Regrouper les deux tables
           -> Mettre en forme les données
----------------------------------------------------------------------------
"""
print("------ Partie 0 ------\n")
rating_cols=["userId","movieId","rating"]
movies_cols=["movieId", "title", "genres"]
ratings=pd.read_csv("./dataset_movies_moodwork/ratings.csv",sep=",",names=rating_cols,usecols=range(3))
movies=pd.read_csv("./dataset_movies_moodwork/movies.csv",sep=",",names=movies_cols,usecols=range(3))
#inner join des deux tables (colonne commune = "movieId")
ratings=pd.merge(movies,ratings,'inner')
#On retire la première ligne car elle contient les intitulés des colonnes
ratings=ratings.ix[1:]
#Le type des notes/ratings est une chaîne de caractère. Pour pouvoir faire des calculs,
#il est nécessaire de la transformer en données numériques (float)
ratings['rating']=pd.to_numeric(ratings['rating'])

#Les genres sont entassés dans une même chaîne de caractères, il faut les séparer et les mettre
#dans une liste (pour chaque utilisateur)
def parseGenre(col):
    newCol=[]
    for element in col :
        newCol.append(element.split('|'))
    return newCol
ratings["genres"]=parseGenre(ratings["genres"])

usersNumber=len(ratings["userId"].unique()) #nombre d'utilisateurs
moviesNumber=len(ratings["movieId"].unique()) #nombre de films
print("Les données contiennent "+str(usersNumber)+" utilisateurs uniques et "+str(moviesNumber)+" films uniques.")
print("Aperçu des données après fusion : ")
print(str(ratings.head()))


"""
----------------------------------------------------------------------------
Partie 1 : -> Question 1
                -> Calculer le nombre de notes par utilisateur
                -> Calculer des indicateurs statistiques (moyenne, min, max...)
                -> Méthode 1 : calcul du nombre d'utilisateurs par groupe
                -> Méthode 2 : calcul du nombre d'utilisateurs par groupe
----------------------------------------------------------------------------
"""
print("\n------ Partie 1 ------\n")
#On affiche le nombre d'utilisateurs par groupe
# Variables : "groupxy" => x : numéro de la méthode (1 ou 2)
#                        y : numéro du groupe (1, 2, 3 ou 4)
usersVotesFrequency=ratings.groupby(["userId"])["rating"].agg(['count','mean'])
print("Aperçu des notes des utilisateurs (nombre de notes + moyenne des notes)")
print(str(usersVotesFrequency.head()))
avgRating=np.mean(usersVotesFrequency['count'])
minRating=np.min(usersVotesFrequency['count'])
maxRating=np.max(usersVotesFrequency['count'])
print("Moyenne du nombre de notes par utilisateur : "+str(avgRating))
print("Nombre minimum de notes attribuées par un même utilistauer : "+str(minRating))
print("Nombre maximum de notes attribuées par un même utilisateur : "+str(maxRating))
print("\nRépartition des utilisateurs en groupes...\nMéthode 1 (voir la doc) :")
#Méthode 1
group11=len([count for count in usersVotesFrequency['count'] if count < 0.25*maxRating])
group12=len([count for count in usersVotesFrequency['count'] if count > 0.25*maxRating and count < 0.5*maxRating])
group13=len([count for count in usersVotesFrequency['count'] if count > 0.5*maxRating and count < 0.75*maxRating])
group14=len([count for count in usersVotesFrequency['count'] if count > 0.75*maxRating])
print("\tGroupe 1 : "+str(group11)+" utilisateurs.")
print("\tGroupe 2 : "+str(group12)+" utilisateurs.")
print("\tGroupe 3 : "+str(group13)+" utilisateurs.")
print("\tGroupe 4 : "+str(group14)+" utilisateurs.")
print("\nMéthode 2 (voir la doc) :")
#Méthode 2
group21=len([count for count in usersVotesFrequency['count'] if count < 0.5*avgRating])
group22=len([count for count in usersVotesFrequency['count'] if count > 0.5*avgRating and count < avgRating])
group23=len([count for count in usersVotesFrequency['count'] if count > avgRating and count < 2*avgRating])
group24=len([count for count in usersVotesFrequency['count'] if count > 2*avgRating])
print("\tGroupe 1 : "+str(group21)+" utilisateurs.")
print("\tGroupe 2 : "+str(group22)+" utilisateurs.")
print("\tGroupe 3 : "+str(group23)+" utilisateurs.")
print("\tGroupe 4 : "+str(group24)+" utilisateurs.")

"""
----------------------------------------------------------------------------
Partie 2 : -> Proposer N films à un nouvel utilisateur
                -> Sélectionner les films les plus populaires
                -> Définir un score pour les films les plus populaires
                -> Créer une fonction qui applique le score et qui
                   retourne les N films avec le plus grand score
----------------------------------------------------------------------------
"""
print("\n------ Partie 2 ------\n")
#Créer une dataframe qui montre -> le film
#                               -> le nombre de ses notes
#                               -> la moyenne de ses notes
movieStats = ratings.groupby('title').agg({'rating': [np.size,np.mean]})
print("Aperçu des statistiques d'un film : nombre de notes et moyenne des notes par film ->")
print(str(movieStats.head()))
avgNbRates=np.mean(movieStats["rating"]["size"]) #10
print("Un film a en moyenne "+str(avgNbRates)+" notes")
globalAverageRating=np.mean(ratings["rating"]) # = 3.5
print("Note moyenne d'un film = "+str(globalAverageRating))
#Sélectionner les films populaires -> ceux qui ont été notés (et donc regardés)
#plus que 90% des autres films. La moyenne étant de 10 notes par film,
#le quantile de 0.9 correspond à sélectionner les films qui ont au moins 27 votes
minNbRates=movieStats["rating"]["size"].quantile(0.90) # = 27
popularMovies = movieStats.copy().loc[movieStats["rating"]["size"] >= minNbRates]
print("On se retrouve avec "+str(np.size(popularMovies))+" films populaires")

def WeightedRating(x,m=minNbRates, C=globalAverageRating): #équation du score (voir doc)
    v=x["rating"]['size'] #nombre de votes du film
    R=x["rating"]['mean'] #note moyenne du film
    return (R* v/(v+m))+(C*m/(v+m))

def recommand_new_user(popularMovies, N):
    # création du score pour chaque film dans une nouvelle colonne
    popularMovies['score']=popularMovies.apply(WeightedRating,axis=1)
    # On ordonne les films par ordre de score décroissant, et on renvoie le top N sous forme d'une liste
    popularMovies = popularMovies.sort_values('score',ascending=False)
    return list(popularMovies.head(N).axes[0])

print("\n\tTest : Recommander 5 films à un nouvel utilisateur.\nRésultat :")
print(str(recommand_new_user(popularMovies,5)))

"""
----------------------------------------------------------------------------
Partie 3 :  -> Proposer N films à un utilisateur dont on possède l'historique
                -> Dresser les notes que chaque utilisateur a attribué à chaque film
                -> Construire une matrice de corrélation entre les films
                -> Pour l'utilisateur à qui on veut recommander des films :
                    -> On dresse la liste des films qu'il a noté
                    -> Pour chacun de ces films :
                        -> On pondère la corrélation avec les autres films par la note
                        qu'a attribué l'utilisateur au film, pour que les films y ressemblant
                        aient la même importance : obtention d'un score
                        -> On rajoute les films résultant à notre liste de recommandations
                    -> On ordonne la liste de recommandations par ordre décroissant
                    et on retourne les N premiers                    
----------------------------------------------------------------------------
"""
print("\n------ Partie 3 ------\n")
#On crée une table de pivot qui permet de connaître la note qu'a attribué
#chaque utilisateur à chaque film. Si l'utilisateur n'a pas noté le film,
#on y retrouve la valeur NaN
pivotTab=ratings.pivot_table(index=['userId'],columns=['title'],values="rating")
print("Aperçu de la table de pivot (NaN <=> l'utilisateur n'a pas noté ce film) : ")
print(str(pivotTab.head(10)))
#Calcul de la matrice de corrélation entre films à partir de la table de pivot
corrMatrix = pivotTab.corr()
print("Aperçu de la matrice de corrélation : ")
print(str(corrMatrix.head()))

def recommand_to_user(userId,pivotTab,corrMatrix,N):
    #On commence par lister les films qu'a regardé l'utilisateur + la
    #note qu'il leur a attribuée
    userRatings=pivotTab.loc[userId].dropna()
    Similars=pd.Series()
    for i in range(len(userRatings.index)): #pour chacun de ces films
        if userRatings[i] > 2.5 : #on ne cherchera des films similaires que s'il a bien noté le film
            sims= corrMatrix[userRatings.index[i]].dropna() #on tire la corrélation avec les autres films
            sims= sims.map(lambda x : x * userRatings[i]) #on la pondère par la note du film avec lequel on compare
            Similars=Similars.append(sims) #On les rajoute à notre liste de recommandations
    Similars.groupby(Similars.index).sum() #Si un film est recommandé plusieurs fois, on somme ses scores
    Similars.sort_values(inplace=True, ascending=False) #On organise par ordre décroissant du score
    for val in userRatings.index : #Si on est sur le point de recommander un film qui l'a déjà vu, on retire
        if val in Similars.index :
            Similars = Similars.drop(val)
    return list(Similars.head(N).axes[0])

print("\n\tTest : recommandation de 5 films à l'utilisateur dont l'idUser=220")
print(str(recommand_to_user("8",pivotTab,corrMatrix,5)))
print("\nSachant que les films qu'il a regardé et la note qu'il leur a attribuée sont les suivants : ")
userRatings=pivotTab.loc["8"].dropna()
userRatings.sort_values(inplace=True,ascending=False)
print(str(userRatings.head(20)))