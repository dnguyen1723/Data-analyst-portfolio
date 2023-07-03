# Projet 10 : Détectez des faux billets
-----
## 1 Importation des librairies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

-----
## 2 Chargement des données

data = pd.read_csv('billets.csv', sep = ';', decimal = '.')
data.head()

On affiche le nombre de lignes et de colonnes.

data.shape

-----
## 3 Nettoyage et préparation des données
### 3.1 Vérification du types de données

data.dtypes

Pas d'erreurs

### 3.2 Vérification des doublons

data.duplicated().sum()

### 3.3 Vérification des valeurs manquantes

data.isnull().sum()

37 Valeurs manquantes dans la colonnes "margin_low".

On fait une régression linéaire multiple pour calculer les valeurs manquantes.

### 3.4 Régression linéaire
On importe d'abord les librairies.

import statsmodels.formula.api as smf
from scipy.stats import t, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
from statsmodels.formula.api import ols
import statsmodels.api as sm

reg_multi = smf.ols('margin_low ~ is_genuine + diagonal + height_left + height_right + margin_up + length', data = data).fit()
print (reg_multi.summary())

Certains paramètres ne sont pas significativement différents de 0, car leur p-valeur n'est pas inférieure à 5 %, le niveau de test que nous souhaitons.

On retire les variables non significatives. On commence par la moins significative 'diagonal' car sa p-value = 0.71

reg_multi = smf.ols('margin_low ~ is_genuine + height_left + height_right + margin_up + length', data = data).fit()
print (reg_multi.summary())

C'est maintenant 'length', avec une p-valeur de 0.88, qui est la moins significative. On la retire.

reg_multi = smf.ols('margin_low ~ is_genuine + height_left + height_right + margin_up', data = data).fit()
print (reg_multi.summary())

On retire 'height_right' qui a une p-valeur de 0.5.

reg_multi = smf.ols('margin_low ~ is_genuine + height_left + margin_up', data = data).fit()
print (reg_multi.summary())

On retire 'height_left' qui a une p-valeur de 0.45.

reg_multi = smf.ols('margin_low ~ is_genuine + margin_up', data = data).fit()
print (reg_multi.summary())

Tous les paramètres sont significatifs (p-values proches de 0). Quant au $R^{2}$, il vaut environ 0.62, tout comme le $R^{2}$ ajusté. On peut utiliser ce modèle à des fins de prévision.

# On crée un dataframe avec les 2 paramètres
model_var = data[['is_genuine', 'margin_up']].copy()

# On crée la variable de la prédiction via la méthode 'predict'
margin_low_prev = reg_multi.predict(model_var)

# On remplace les valeurs manquantes par la valeur prédite
data['margin_low'].fillna(margin_low_prev, inplace = True)

# On vérifie les valeurs manquantes
data.isnull().sum()

Nous n'avons plus de valeurs manquantes.

-----
## 4 Analyse exploratoire : vrais VS faux billets
On convertit les valeurs booléennes en string pour utiliser des fonctions.

str_genui_data = data.copy()
str_genui_data['is_genuine'] = str_genui_data['is_genuine'].map({True: 'True', False : 'False'})
str_genui_data.dtypes

### 4.1 Diagonale

sns.set()

plt.figure(figsize=(10, 2))
sns.boxplot(x= "diagonal", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Diagonale en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

### 4.3 Hauteur gauche

plt.figure(figsize=(10, 2))
sns.boxplot(x= "height_left", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Hauteur gauche en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

### 4.4 Hauteur droite

plt.figure(figsize=(10, 2))
sns.boxplot(x= "height_right", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Hauteur droite en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

### 4.5 Marge du bas

plt.figure(figsize=(10, 2))
sns.boxplot(x= "margin_low", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Marge bas en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

### 4.6 Marge haut

plt.figure(figsize=(10, 2))
sns.boxplot(x= "margin_up", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Marge haut en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

### 4.7 Longueur

plt.figure(figsize=(10, 2))
sns.boxplot(x= "length", 
            y= "is_genuine",
            data= str_genui_data);
plt.xlabel('Longueur en mm', fontsize = 14)
plt.ylabel('Type de billets', fontsize = 14)

-----
## 5 Algorithmes de détection
### 5.1 Régression logistique
On effectue une régression logistique de 'is_genuine'.

On importe d'abord les librairies.

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

#### 5.1.1 Préparation des données
On crée un dataframe où les valeurs de 'is_genuine' sont convertis en nombres entiers.

int_genui_data = str_genui_data.copy()
int_genui_data['is_genuine'] = int_genui_data['is_genuine'].map({'True': 0, 'False' : 1})
int_genui_data.dtypes

On sépare x, la variable à expliquer et y, les variables explicatives.

X = int_genui_data.drop(columns = "is_genuine")
y = int_genui_data.is_genuine

#### 5.1.2 Partage de l'échantillon pour le test et l'entraînement
On spécifie le % de données que nous voulons par rapport au train (20 % dans le test et 80 % dans le train) et le random_state (la fonction s'éxécute toujours de la même façon et passe toujours le même split).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# On vérifie leur forme
print (f"Nous avons X_train de forme {X_train.shape} et y_train de forme {y_train.shape} ")
print (f"Nous avons X_test de forme {X_test.shape} et y_test de forme {y_test.shape} ")

#### 5.1.3 Entraînement du modèle
On instancie un estimateur, c'est-à-dire l'algorithme qui va permettre de comprendre quel modèle utiliser dans les données.

estimator = LogisticRegression(solver = 'liblinear')

# On entraîne l'estimateur
estimator.fit(X_train, y_train)

On calcule notre vecteur de prédiction.

y_pred = estimator.predict(X_test)
y_pred[:10]

La régression logistique (estimator = LogisticRegression(solver = 'liblinear')) peut également nous permettre d'évaluer la probabilité d'appartenance à une classe.

On appelle la méthode predict_proba.

y_prob = estimator.predict_proba(X_test).round(2)
y_prob[:10]

On a une matrice à 2 colonnes où chaque colonne représente la probabilité/confiance du modèle que tel individu appartient à telle classe.

Exemple : pour le 1er individu, le modèle estime à 0% la probabilité d'appartenir à la classe 0 et 100% la probabilité d'appartenir à la classe 1.

Pour le 2ème individu, c'est 0.1% d'appartenance à la classe 0 et 0.99% d'appartenance à la classe 1.

On calcule 2 scores: la performance de l'estimateur sur les données de train et de test.

tr_score = estimator.score(X_train, y_train).round(4)
te_score = estimator.score(X_test, y_test).round(4)

print (f"score train : {tr_score} score test : {te_score}")

Nos scores ont 99% d'accuracy sur le train et sur le test. Nos modèles fonctionnent bien.

En général, le score de train est meilleur que le score de test.

D'autres scores seront calculés (pour le k-means), on fait une fonction score.

# la fonction s'appelle score et prend en argument un estimateur
def score(estimator) :
    # les 3 guillements référencent une docstring et laisse une indication sur ce que fait la fonction
    """compute and print train scrore and test score"""
    
    # On copie-colle la cellule précédente
    tr_score = estimator.score(X_train, y_train).round(4)
    te_score = estimator.score(X_test, y_test).round(4)
    
    print (f"score train : {tr_score} score test : {te_score} ")

On teste la fonction.

score(estimator)

#### 5.1.4 Test du modèle
On veut tester la qualité de notre modèle en comparant les données réelles pour une variable cible à celles prédites par le modèle.

On utilise une matrice de confusion (tableau de contingence) qu'on appelle sur y_test et y_pred et on regarde les différences entre les valeurs pred_1 (en réalité 0) ou les valeurs pred_0 (en réalité 1).

mat = confusion_matrix(y_test, y_pred)
mat

On transforme la matrice en dataframe et on renomme les colonnes et les index.

mat = pd.DataFrame(mat)
mat.columns = [f"pred_{i}" for i in mat.columns]
mat.index = [f"test_{i}" for i in mat.index]
mat

La matrice nous montre que:
- sur la valeur 0, on a prédit 190 fois la valeur 0 et 0 fois la valeur 1;
- sur la valeur 1, on a prédit 3 fois la valeur 0 et 107 fois la valeur 1.

D'autres matrices de confusion seront utilisées (pour le k-means), on créé une fonction 'confusion'.

def confusion (y_test, y_pred) :
    """ display a fancy confusion matrix"""
    
    mat = confusion_matrix(y_test, y_pred)
    mat = pd.DataFrame(mat)
    mat.columns = [f"pred_{i}" for i in mat.columns]
    mat.index = [f"test_{i}" for i in mat.index]
    
    return mat

On teste la fonction.

confusion(y_test, y_pred)

On calcule le rappel, la précision, le F1 score...

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate).round(2)
print(roc_auc)

plt.figure(figsize = (10, 6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, color ='red', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc= 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### 5.2 K-means
On importe d'abord les librairies.

from sklearn.cluster import KMeans

#### 5.2.1 Nombre de clusters
On définit le nombre de clusters par la méthode du coude.

# On stocke nos inerties
intertia = []

# On définit la liste du nombre de clusters que l'on veut tester, soit 10
k_list = range(1, 10)
list(k_list)

# Pour chaque valeur de k, on entraine un k-means spécifique et on stocke son inertie
for i in k_list :
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X) 
    intertia.append(kmeans.inertia_)

# La liste d'inerties
intertia

On affiche le graphique de l'inertie intraclasse selon le nombre de clusters.

fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("Inertie", fontsize = 14)
ax.set_xlabel("Nombre de clusters", fontsize = 14)
ax = plt.plot(k_list, intertia)

On a une cassure à 2 clusters.
#### 5.2.2 Clusters

# On entraîne l'estimateur
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train, y_train)

# On stocke les labels dans une variable
labels = kmeans.labels_
labels

#### 5.2.3 Entraînement du modèle
On fait une prédiction sur X_test.

prediction = kmeans.predict(X_test)

pred_df = pd.DataFrame({'actual' : y_test, 'prediction' : prediction})

print (pred_df)

# On réinitialise l'index d'y_test pour pouvoir faire des calculs sur le nombre d'individus et la précision
y_test = y_test.reset_index(drop=True)

correct = 0

for i in range(len(y_test)):
    if prediction[i] == y_test[i]:
        correct += 1
        
print(correct/len(y_test))

Nous avons 98% d'accuracy.
#### 5.2.4 Test du modèle
On fait une matrice de confusion qu'on apelle sur les valeurs réelles 'y' et et les valeurs prédites 'prediction'.

kmeans_mat = confusion_matrix(y_test, prediction)
kmeans_mat

On transforme la matrice en dataframe et on renomme les colonnes et les index.

kmeans_mat = pd.DataFrame(kmeans_mat)
kmeans_mat.columns = [f"pred_{i}" for i in kmeans_mat.columns]
kmeans_mat.index = [f"test_{i}" for i in kmeans_mat.index]
kmeans_mat

Les résultats sont légèrements moins bons que ceux de la régression logistique.

-----
## 5 Application du modèle
On applique le modèle choisi (régression logistique) sur le dataset 'billets_productions'.

On charge les données.

prod_data = pd.read_csv('billets_production.csv')
print (prod_data.shape)
prod_data.head()

On supprime la colonne 'id' pour pouvoir utiliser notre modèle.

prod_data = prod_data.drop(columns = 'id')
prod_data

On applique notre modèle sur les données.

prod_genui = estimator.predict(prod_data)
prod_genui

On ajoute nos résultats au dataframe.

# On crée une variable des résultats
prod_data["is_genuine"] = prod_genui

# On convertit les résultats en valeur booléenne
prod_data.is_genuine = ~prod_data.is_genuine.astype('bool')

prod_data

Sur les 5 billets, d'après notre modèle nous avons 3 faux billets.