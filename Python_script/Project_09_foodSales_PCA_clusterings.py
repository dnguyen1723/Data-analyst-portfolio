# Projet 9 : La poule qui chante
## Partie 2 : Clusterings, analyse des centroïdes et ACP
-----
## 1 Importation des librairies

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestCentroid

On charge Seaborn

sns.set()

-----
## 2 Importation des données

df_merged = pd.read_csv('p9_merged.csv')
df_merged.head()

Suite à une 1ère analyse, la Chine et l'Inde ne nous permettent pas de visualiser correctement certaines données, comme par exemple la variance de la population des clusters, leur population étant beaucoup plus importante que la moyenne mondiale.

Afin d'avoir une analyse plus fine et pertinente, nous supprimons ces 2 pays du dataset.

df_merged.drop(df_merged[df_merged['Zone'] == 'Chine, continentale'].index, inplace = True)
df_merged.drop(df_merged[df_merged['Zone'] == 'Inde'].index, inplace = True)



-----
## 3 Clusterings
### 3.1 Classification ascendante hiérarchique
#### 3.1.1 Dendrogramme
On calcule nos distances avec la méthode Ward et la librairie scikit-learn. Cette matrice de distance sera notée Z.

# Mettre la variable 'Zone' en tant qu'index
df_merged = df_merged.set_index('Zone')

# Calcul de la matrice de distance avec la méthode Ward
Z = linkage(df_merged, method = "ward")
pd.DataFrame(Z).head()

On affiche notre dendrogramme.

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

_ = dendrogram(Z, ax=ax)

plt.title("Classification hiérarchique", fontsize = 14)
plt.ylabel("Distance", fontsize = 14)
plt.tick_params(axis ='x', labelbottom = False)
plt.show()

Sur l'axe y, la distance entre clusters s'agrandit fortement à partir d'une distance de 20 000. Nous choisissons de grouper nos individus en 5 clusters. 

# les arguments p=5, truncate_mode="lastp" vont afficher 5 clusters

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

_ = dendrogram(Z, p=5, truncate_mode = "lastp", ax=ax)

plt.title("Classification hiérarchique", fontsize = 14)
plt.xlabel("Nombre de points", fontsize = 14)
plt.ylabel("Distance", fontsize = 14)
plt.show()

#### 3.1.2 Définition des clusters avec scikit-learn

# On instancie un estimateur
cah = AgglomerativeClustering(n_clusters=5, linkage="ward")

# On entraîne l'estimateur
cah.fit(df_merged)

# On affiche les clusters
cah.labels_

# On remplace les nombres par des lettres pour nommer les clusters
dd = {i:j for i,j in enumerate(list("abcde"))}
labels = [dd[i] for i in cah.labels_]
labels[:10]

# On ajoute les labels au dataframe
df_hierarchical = df_merged.copy()
df_hierarchical["cah_cluster"] = labels
df_hierarchical.head()

#### 3.1.3 Calcul des centroïdes

# On stocke les centroïdes dans une variable
z_predict = cah.fit_predict(Z)
clf = NearestCentroid()
clf.fit(Z, z_predict)
clf.centroids_

### 3.2 K-means

#### 3.2.1 Définition du nombre de clusters : méthode du coude

# On stocke nos inerties
intertia = []

# On définit la liste du nombre de clusters que l'on veut tester
k_list = range(2, 10)
list(k_list)

# Pour chaque valeur de k, on entraine un k-means spécifique et on stocke son inertie
for i in k_list :
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(df_merged) 
  intertia.append(kmeans.inertia_)

# La liste d'inerties
intertia

# Graphique inertie intraclasse

fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("Inertie", fontsize = 14)
ax.set_xlabel("Nombre de clusters", fontsize = 14)
ax = plt.plot(k_list, intertia)

L'inertie intraclasse diminue fortement à partir de 5 et 6 clusters. Nous choisissons un nombre de 5 clusters pour faciliter la comparaison des 2 méthodes de clustering.
#### 3.2.2 Clusters

# On entraîne l'estimateur
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_merged)

kmeans.labels_

# On remplace les chiffres par des lettres
dd = {i:j for i,j in enumerate(list("abcde"))}
dd

labels = [dd[i] for i in kmeans.labels_]
labels[:10]

# On ajoute nos labels à df_kmeans
df_kmeans = df_merged.copy()
df_kmeans["kmeans_cluster"] = labels
df_kmeans.head()

#### 3.2.3 Calcul des centroïdes

# On stocke les centroïdes dans une variable
centroids = kmeans.cluster_centers_
centroids

-----
## 3 Comparaison classification hiérarchique VS K-means
### 3.1 Liste des pays des clusters
On compare la correspondance des clusters de chaque méthode.

# On réindexe les 3 dataframes
df_merged = df_merged.reset_index()
df_hierarchical = df_hierarchical.reset_index()
df_kmeans = df_kmeans.reset_index()

# On ajoute les colonnes 'cah_cluster' et 'kmeans_cluster' au dataframe 'df_merged'
df_merged = df_merged.merge(df_hierarchical[['Zone', 'cah_cluster']])
df_merged = df_merged.merge(df_kmeans[['Zone', 'kmeans_cluster']])

On modifie les labels de 'kmeans_cluster' afin qu'ils correspondent à ceux de 'cah_cluster'.

# On vérifie les correspondances du cluster classification 'a'

df_merged.loc[df_merged['kmeans_cluster'] == 'a', ('Zone', 'cah_cluster', 'kmeans_cluster')].head()

# On vérifie les correspondances du cluster classification 'b'

df_merged.loc[df_merged['kmeans_cluster'] == 'b', ('Zone', 'cah_cluster', 'kmeans_cluster')].head()

# On vérifie les correspondances du cluster classification 'c'

df_merged.loc[df_merged['kmeans_cluster'] == 'c', ('Zone', 'cah_cluster', 'kmeans_cluster')].head()

# On vérifie les correspondances du cluster classification 'd'

df_merged.loc[df_merged['kmeans_cluster'] == 'd', ('Zone', 'cah_cluster', 'kmeans_cluster')].head()

# On vérifie les correspondances du cluster classification 'e'

df_merged.loc[df_merged['kmeans_cluster'] == 'e', ('Zone', 'cah_cluster', 'kmeans_cluster')].head()

# On remplace les labels des kmeans_clusters

df_merged['kmeans_cluster'].replace((['a', 'b', 'c', 'e']), ['e', 'c', 'a', 'b'], inplace = True)

df_merged[['Zone', 'cah_cluster', 'kmeans_cluster']].head(10)

On calcule le taux de correspondance des 2 méthodes de clustering.

# On affiche le nombre total de pays
print (len(df_merged.index))

# On affiche le nombre de pays qui font partis des mêmes clusters des 2 méthodes de clustering
print (len(df_merged.query('cah_cluster == kmeans_cluster')))

# On affiche le taux de correspondance
print (len(df_merged.query('cah_cluster == kmeans_cluster')) / len(df_merged.index) * 100)

#### 150 pays sur 168 font partis des même clusters sur les 2 méthodes de clustering, soit 90 % des pays.
#### Les clusters des 2 méthodes sont très similaires.

df_merged.loc[df_merged['kmeans_cluster'] == 'a', :]

### 3.2 Variance
#### 3.2.1 Disponibilité volailles classification hiérarchique

# On trie par ordre alphabétique des clusters
cah_cluster_sorted = sorted(df_merged.cah_cluster.unique())

plt.figure(figsize=(10,6))
sns.boxplot(x="Disponibilité alimentaire en quantité (kg/personne/an)", 
            y="cah_cluster",
            order = cah_cluster_sorted,
            data=df_merged);
plt.title("Variance de la disponibilité volailles des 'cah_cluster'", fontsize = 14)
plt.xlabel('Quantité en Kg/pers/an', fontsize = 14)
plt.ylabel('cah_cluster', fontsize = 14)

#### 3.2.2 Disponibilité volailles K-means

# On trie par ordre alphabétique des clusters
kmeans_cluster_sorted = sorted(df_merged.kmeans_cluster.unique())

plt.figure(figsize=(10,6))
sns.boxplot(x="Disponibilité alimentaire en quantité (kg/personne/an)", 
            y="kmeans_cluster",
            order = kmeans_cluster_sorted,
            data=df_merged);
plt.title("Variance de la disponibilité volailles des 'kmeans_cluster'", fontsize = 14)
plt.xlabel('Quantité en Kg/pers/an', fontsize = 14)
plt.ylabel('kmeans_cluster', fontsize = 14)

#### 3.2.3 Population classification hiérarchique

plt.figure(figsize=(10,6))
sns.boxplot(x="Population en milliers", 
            y="cah_cluster",
            order = cah_cluster_sorted,
            data=df_merged);
plt.title("Variance de la population des 'cah_cluster'", fontsize = 14)
plt.xlabel('Valeur en milliers', fontsize = 14)
plt.ylabel('cah_cluster', fontsize = 14)

#### 3.2.4 Population K-means

plt.figure(figsize=(10,6))
sns.boxplot(x="Population en milliers", 
            y="kmeans_cluster",
            order = kmeans_cluster_sorted,
            data=df_merged);
plt.title("Variance de la population des 'kmeans_cluster'", fontsize = 14)
plt.xlabel('Valeur en milliers', fontsize = 14)
plt.ylabel('kmeans_cluster', fontsize = 14)

#### 3.2.5 PIB par habitant classification hiérarchique

plt.figure(figsize=(10,6))
sns.boxplot(x="PIB/hab en $", 
            y="cah_cluster",
            order = cah_cluster_sorted,
            data=df_merged);
plt.title("Variance du PIB par habitant des 'cah_cluster'", fontsize = 14)
plt.xlabel('Valeur en dollars', fontsize = 14)
plt.ylabel('cah_cluster', fontsize = 14)

#### 3.2.6 PIB par habitant K-means

plt.figure(figsize=(10,6))
sns.boxplot(x="PIB/hab en $", 
            y="kmeans_cluster",
            order = kmeans_cluster_sorted,
            data=df_merged);
plt.title("Variance du PIB par habitant des 'kmeans_cluster'", fontsize = 14)
plt.xlabel('Valeur en dollars', fontsize = 14)
plt.ylabel('kmeans_cluster', fontsize = 14)

-----
## 4 Analyse des centroïdes

# On crée un dataframe
centroids_df = pd.DataFrame(centroids, columns = ['Disponibilité alimentaire (Kcal/personne/jour)', 'Disponibilité alimentaire en quantité (kg/personne/an)', 'Disponibilité de matière grasse en quantité (g/personne/jour)', 'Disponibilité de protéines en quantité (g/personne/jour)', 'Disponibilité intérieure', 'Importations - Quantité', 'Nourriture', 'Production', 'Variation de stock', 'Population en milliers', 'PIB/hab en $'])

# On instancie notre scaler
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler()

# On l'entraîne
std_scale.fit(centroids_df)

# On transforme nos centroïdes
centroids_scaled = std_scale.transform(centroids_df)

# On vérifie le centrage et la réduction
pd.DataFrame(centroids_scaled).describe().round(2).iloc[1:3:, : ]

# On convertit 'centroids_scaled' en dataframe
centroids_scaled_df = pd.DataFrame(centroids_scaled, columns = ['Disponibilité alimentaire (Kcal/personne/jour)', 'Disponibilité alimentaire en quantité (kg/personne/an)', 'Disponibilité de matière grasse en quantité (g/personne/jour)', 'Disponibilité de protéines en quantité (g/personne/jour)', 'Disponibilité intérieure', 'Importations - Quantité', 'Nourriture', 'Production', 'Variation de stock', 'Population en milliers', 'PIB/hab en $'])

# On affiche la heatmap
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(centroids_scaled_df, vmin=-1, annot = True, cmap="coolwarm", fmt = "0.2f")

#### La population contribue majoritairement dans le clustering et le PIB par habitant dans une moindre mesure.
-----
## 5 Analyse des composantes principales
### 5.1 Sélection des variables
On supprime les colonnes inutiles :
- 'cah_cluster' et 'kmeans_cluster'
- Au niveau des disponibilités alimenaires, on ne garde que la disponibilité en quantité (kg/pers/an)
- La disponibilité intérieure, la nourriture et la production sont similaires. On ne garde que la disponibilité intérieure

pca_df = df_merged.copy()
pca_df.drop(['cah_cluster',
             'kmeans_cluster',
             'Disponibilité alimentaire (Kcal/personne/jour)',
             'Disponibilité de matière grasse en quantité (g/personne/jour)', 
             'Disponibilité de protéines en quantité (g/personne/jour)',
             'Nourriture', 
             'Production'], axis = 1, inplace = True)

# On met la variable 'Zone' en tant qu'index
pca_df = pca_df.set_index('Zone')

### 5.2 Préparation des données
On sépare nos données.

X = pca_df.values
X[:2]

On vérifie le type de nos données.

type(X)

On vérifie la forme de la matrice.

X.shape

On enregistre les pays dans une variable 'zones' et nos colonnes dans 'features'.

zones = pca_df.index
features = pca_df.columns

print(zones)
print(features)

### 5.3 Scaling

# On instancie
scaler = StandardScaler()

# On fit
scaler.fit(X)

# On transforme
X_scaled = scaler.transform(X) # On peut faire les 2 opérations en une seule : X_scaled = scaler.fit_transform(X)

# On calcule la moyenne (en espérant qu'elle soit = 0) et l'écart-type (en espérant qu'il soit = 1)
idx = ['mean', 'std']

pd.DataFrame(X_scaled).describe().round(2).loc[idx, :]

### 5.4 ACP
On travaille sur toutes les composantes.

n_components = 6

# On instancie notre ACP
pca = PCA(n_components=n_components)

# On l'entraîne sur les données scalées
pca.fit(X_scaled)

### 5.5 Variance captée et diagramme d'éboulis
On regarde la variance captée (inertie cumulée) pour chaque nouvelle composante.

pca.explained_variance_ratio_

La 1ère composante capte 36 % de la variance de nos données initiales, la 2ème 22 % etc.

# On enregistre ces valeurs dans une variable
scree = (pca.explained_variance_ratio_*100).round(2)

# On crée une variable des variances cumulées
scree_cum = scree.cumsum().round()
scree_cum

# On définit une variable avec la liste de nos composantes
x_list = range(1, n_components+1)
list(x_list)

# On affiche le graphique
plt.figure(figsize = (10, 6))
plt.bar(x_list, scree)
plt.plot(x_list, scree_cum,c="red",marker='o')
plt.xlabel("Rang de l'axe d'inertie", fontsize = 14)
plt.ylabel("pourcentage d'inertie", fontsize = 14)
plt.title("Eboulis des valeurs propres", fontsize = 14)
plt.show(block=False)

Les 2 premières composantes cumulées représentent 58 % de la variance, les 3 premières 76 % et les 4 premières 87 %.
### 5.6 Composantes

# On crée une variable des composantes
pcs = pca.components_

# On la convertit en dataframe
pcs = pd.DataFrame(pcs)

# On renomme les colonnes
pcs.columns = features

# On renomme les indexs
pcs.index = [f"F{i}" for i in x_list]
pcs.round(2).head()

# On pivote 'pcs' pour une meilleure lisibilité
pcs.T

# On affiche une heatmap
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(pcs.T, vmin=-0.1, annot = True, cmap="coolwarm", fmt = "0.2f")
ax.tick_params(axis='y', which='major', labelsize=18)

### 5.7 Cercle des corrélations

# On définit nos axes x et y et on utilise nos 2 premières composantes
x, y = 0, 1

# On affiche le graphique
fig, ax = plt.subplots(figsize=(10, 9))
for i in range(0, pca.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  #0 for PC1
             pca.components_[1, i],  #1 for PC2
             head_width=0.07,
             head_length=0.07, 
             width=0.02,              )

    plt.text(pca.components_[0, i] + 0.05,
             pca.components_[1, i] + 0.05,
             features[i])
    
# affichage des lignes horizontales et verticales
plt.plot([-1, 1], [0, 0], color='grey', ls='--')
plt.plot([0, 0], [-1, 1], color='grey', ls='--')

# nom des axes, avec le pourcentage d'inertie expliqué
plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)), fontsize = 14)
plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)), fontsize = 14)

plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1), fontsize = 14)

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plt.axis('equal')
plt.show(block=False)

# On en fait une fonction

def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)), fontsize = 14)
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)), fontsize = 14)

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1), fontsize = 14)

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)

# On vérifie la fonction pour F1 et F2 en reprécisant 0 et 1

x_y = (0,1)
correlation_graph(pca, x_y, features)

# On applique la fonction pour F3 et F4

correlation_graph(pca, (2,3), features)

### 5.8 Projection des points
On calcule les coordonnées de nos points dans le nouvel espace.

X_proj = pca.transform(X_scaled)
X_proj[:2]

#### On affiche nos points

# On remplace les labels par des nombres pour colorier les clusters 
for i in range(len(labels)):
 
    # replace 'a' with 0
    if labels[i] == 'a':
        labels[i] = 0
        
    # replace 'b' with 1
    if labels[i] == 'b':
        labels[i] = 1
        
        # replace 'b' with 2
    if labels[i] == 'c':
        labels[i] = 2
        
        # replace 'b' with 3
    if labels[i] == 'd':
        labels[i] = 3
        
        # replace 'b' with 4
    if labels[i] == 'e':
        labels[i] = 4
        
# On convertit 'X_proj' en dataframe
X_proj = pd.DataFrame(X_proj, columns = ["F1", "F2", "F3", "F4", "F5", "F6"])

# On garde une version array de 'X_proj'
X_ = np.array(X_proj)

# On affiche la graphique
fig, ax = plt.subplots(1,1, figsize=(10, 8))
ax.scatter(X_proj.iloc[:, 0], X_proj.iloc[:, 1], c= labels, cmap="Set1")

# Valeur x max et y max
x_max = np.abs(X_[:, x]).max() *1.1
y_max = np.abs(X_[:, y]).max() *1.1

# On borne x et y 
ax.set_xlim(left=-x_max, right=x_max)
ax.set_ylim(bottom= -y_max, top=y_max)

# Affichage des lignes horizontales et verticales
plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

# Titre et display
plt.title(f"Projection des individus (sur F{x+1} et F{y+1})", fontsize = 14)
plt.xlabel("F1", fontsize = 14)
plt.ylabel("F2", fontsize = 14)
plt.show()
plt.show()