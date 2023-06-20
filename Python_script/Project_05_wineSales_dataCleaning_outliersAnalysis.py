## 1. Conversion des fichiers XLSX vers CSV
## 2. Remplacement des point-virgules par des virgules via Notepad
## 3. Importation des librairies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## 4. Importation des fichiers CSVs

erp = pd.read_csv('erp.csv')
liaison = pd.read_csv('liaison.csv')
web = pd.read_csv('web.csv', encoding='latin1')

## 5. Aperçu des dataframes

### 5.1 Aperçu table 'erp'

erp.head()

### 5.2 Aperçu table 'liaison'

liaison.head()

### 5.3 Aperçu table 'web'

web.head()

## 6. Vérification des erreurs
- Vérifier le type de variable
- Vérifier les valeurs manquantes
- Vérifer les doublons

### 6.1 Table 'erp'

#### 6.1a Vérification des types de variables

erp.dtypes

#### Colonnes 'product_id et 'onsale_web' de type int64 -> convertion en object

erp['product_id'] = erp['product_id'].astype(str)
erp['onsale_web'] = erp['onsale_web'].astype(str)

#### 6.1b Vérification des valeurs manquantes

erp.isnull().sum()

#### Aucune valeur manquante

#### 6.1c Vérification des doublons

erp['product_id'].duplicated(keep=False).sum()

#### Aucun doublon

### 6.2 Table 'liaison'

#### 6.2a Vérification des types de variables

liaison.dtypes

#### Colonne 'product_id' de type int64 -> convertion en object

liaison['product_id'] = liaison['product_id'].astype(str)

#### 6.2b Vérification des valeurs manquantes

liaison.isnull().sum()

#### 91 valeurs manquantes dans la colonne 'id_web'.

#### Stocker ces valeurs dans un dataframe 'liaison_na' et, après jointure des tables 'web' et 'liaison', vérifier si ces lignes correspondent à un 'product_id' de la table 'web'.

liaison_na = liaison.loc[liaison['id_web'].isnull(), :]

#### 6.2c Vérification des doublons

liaison['product_id'].duplicated(keep=False).sum()

#### Aucun doublon

### 6.3 Table 'web'

#### 6.3a Vérification des types de variable

web.dtypes

#### Dates au format 'object'. Pas de requêtes concernant les dates -> Ignorer

#### 6.3b Vérification des valeurs manquantes

web.isnull().sum().head(6)

#### 85 valeurs manquantes dans la colonne 'sku' et 83 valeurs manquantes dans la colonne 'total_sales' -> Lignes à vérifier

#### Afficher les lignes dont la valeur 'sku' est manquante et 'total_sales' non manquante

web.loc[(web['sku'].isnull()) & (web['total_sales'].notnull()), ['sku', 'total_sales']]

#### Valeur manquante 'sku' -> 'total_sales' manquant ou 0 -> lignes à supprimer

web.drop(web.loc[web['sku'].isnull(), :].index, inplace = True)

#### 6.3c Vérification des doublons

web['sku'].duplicated(keep=False).sum()

#### 1428 doublons -> compter le nombre de doublons par valeurs 'sku'

web_duplicated = web.pivot_table(index = ['sku'], aggfunc = 'size')
web_duplicated

#### Chaque valeur 'sku' possède un doublon -> chercher la cause

#### Comparaison des colonnes d'un doublon

web.loc[web['sku'] == '10014']

#### -> Colonnes 'tax_status', 'post-type' et 'post_mime_type' différents

#### Suppression des lignes dont la valeur de 'post_type' est 'attachment'

web.drop(web[web['post_type'] == 'attachment'].index, inplace = True)

#### Revérification des doublons

web['sku'].duplicated(keep=False).sum()

#### Aucun doublon

## 7. Requête 1 : rapprochement des tables
### 7.1 Jointure des tables 'web' et 'liaison'

Renommer 'id_web' en 'sku' dans la table 'liaison' pour faciliter la jointure

liaison.rename(columns={'id_web':'sku'}, inplace = True)

Jointure à gauche des tables 'web' et 'liaison' car on garde toutes les clés de la table 'web'

web = pd.merge(web, liaison, on='sku', how='left', indicator = True)

Déplacement de la colonne 'product_id' en 2ème position dans la table 'web'

col = web.pop('product_id')
df = web.insert(1, 'product_id', col)

Aperçu de la table 'web'

web.head()

Vérification de la jointure : si aucune valeur différente de 'both' dans la colonne '_merge' = jointure OK

web.loc[web['_merge'] != 'both']

#### Pas d'erreurs dans la jointure

Valeurs manquantes dans la table 'liaison' : vérifier si les valeurs de la colonne 'product_id' de la table 'liaison_na' correspondent à une valeur 'product_id' de la table 'web'.

Création un dataframe des correspondances des valeurs manquantes

na_corresp = liaison_na.assign(InWeb = liaison_na.product_id.isin(web.product_id).astype(str))
na_corresp.head()

Afficher les correspondances

na_corresp.loc[na_corresp['InWeb'] == 'True']

#### Aucune correspondance -> Valeurs 'id_web' nulles car produits non commercialisés en ligne

### 7.2 Jointure des tables 'web' et 'erp'

Jointure à gauche car on garde toutes les clés de la table 'web'

web = pd.merge(web, erp, on='product_id', how = 'left')

Déplacement des colonnes 'onsale_web', 'price', 'stock_quantity' et 'stock_status' en 3, 4, 5 et 6ème position dans la table 'web'

col_1 = web.pop('onsale_web')
df = web.insert(2, 'onsale_web', col_1)
col_2 = web.pop('price')
df = web.insert(3, 'price', col_2)
col_3 = web.pop('stock_quantity')
df = web.insert(4, 'stock_quantity', col_3)
col_4 = web.pop('stock_status')
df = web.insert(5, 'stock_status', col_4)

Aperçu de la table 'web'

web.head()

Vérification de la jointure : si aucune valeur différente de 'both' dans la colonne '_merge' = jointure OK

web.loc[web['_merge'] != 'both']

#### Pas d'erreurs dans la jointure

## 8. Requête 2 : Chiffre d'affaires
### 8.1 Chiffre d'affaires par produit

Création de la variable 'montant' (='price' x 'total_sales') dans la table 'web'

web['montant'] = web['price'] * web['total_sales']

Agrégation de la variable 'product_id' en additionnant les valeurs de la colonne 'montant'

CA_df = web.groupby(['product_id', 'post_title', 'price', 'onsale_web'])[['montant']].sum()

Réinitialisation de l'index

CA_df.reset_index(inplace=True)

Vérification du nombre d'individus

CA_df.shape

Renommer la colonne 'montant' par 'CA'

CA_df.rename(columns={'montant' : 'CA'}, inplace = True)

Aperçu de la table 'CA_df'

CA_df.head()

#### Représentation graphique

plt.figure(figsize=(15,6))
plt.rcParams.update({'font.size': 14})
sns.set_palette('Set2')
CA_hist = sns.barplot(data=CA_df, x = 'product_id', y = 'CA')
CA_hist.set(xticklabels=[])
CA_hist.tick_params(bottom=False)
CA_hist.set(xlabel='Produit')
plt.title("Chiffre d'affaires par produit")
plt.grid(axis='y')

plt.figure(figsize=(15,6))
plt.rcParams.update({'font.size': 14})
sns.set_palette('Set2')
CA_hist = sns.histplot(data=CA_df, x = 'CA', bins = 100)
CA_hist.set(xlabel="Chiffre d'affaires")
CA_hist.set(ylabel="Effectif")
CA_hist.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000])
CA_hist.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]) 
plt.xticks(rotation=45)
plt.title("Effectif par chiffre d'affaires")
plt.grid(axis='y')

#### La majorité des produits (65%) ont un chiffre d'affaires situé entre 0 et 50 euros.

Afficher les 10 CA les plus élevés

CA_df.nlargest(10, 'CA')

#### 70% des produits au CA le plus élevé ont un valeur située entre 17 et 55 euros.
#### 6 champagnes présents dans la liste.

Afficher le nombre de produits dont le CA = 0

noCA = CA_df.loc[CA_df['CA'] == 0, :].shape[0]
noCA

#### 329 produits dont le CA est égal à 0.

Calcul du taux des produits dont le CA = 0

tx_noCA = round((noCA / len(CA_df)) * 100, 2)
tx_noCA

#### 46 % des produits ont un chiffre d'affaires nul, soit presque 1 produit sur deux.

### 8.2 Chiffre d'affaires réalisé en ligne

CA_online = CA_df.loc[CA_df['onsale_web'] == '1', 'CA'].sum()
CA_online

#### Le chiffre d'affaires réalisé en ligne s'élève à 70 569 euros.

Chiffre d'affaires moyen par produit en ligne.

CA_mean = CA_online / len(CA_df)
CA_mean

#### Le chiffre d'affaires moyen par produit en ligne s'élève à 99 euros.

Calcul du CA moyen des produits en ligne moins les produits CA = 0

CA_mean_2 = CA_online / (len(CA_df) - noCA)
CA_mean_2

#### Le chiffre d'affaires moyen par produit en ligne excluant les produits au CA nul s'élève à 183 euros.

## 9. Analyse prix des produits
### 9.1 Agrégation de la variable 'product_id' dans la table 'web'

product_df = web.groupby(['product_id', 'post_title'])[['price']].mean()

Réinitialisation d'un index

product_df.reset_index(inplace=True)

Vérification nombre individus

product_df.shape

Aperçu de la table 'product_df'

product_df.head()

### 9.2 Distribution empirique et représentation de l'écart-type et des outliers

print('Distribution empirique des prix des produits\n')
print(product_df['price'].describe())
print('\nEcart-type =', product_df['price'].std(ddof=0))
plt.figure(figsize=(15,2))
plt.rcParams.update({'font.size': 14})
product_df.boxplot(column='price', vert=False)
plt.xlabel("Prix des produits en euros")
plt.title('Ecart-type et outliers')
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
plt.show()

### 9.3 Liste des outliers

#### Outlier = Q3 + (1,5 x IQ) = 42,2 + (1,5 x 27,8) = 83,9

Création d'un dataframe des outliers

product_out = product_df[product_df['price'] > 83.9]

Liste des outliers

display(product_out.sort_values(by=['price'], ascending=False))

#### Présence de nombreux grands crus -> valeurs cohérentes.

Calcul du nombre d'outliers et du taux d'outliers par rapport au nombre total de produits

product_out.shape

(len(product_out) / len(product_df)) * 100

#### 32 produits ont une valeur aberrante, soit 4,5% du total des produits.
### 9.4 Représentation graphique des outliers

plt.figure(figsize=(15,4))
plt.rcParams.update({'font.size': 14})
sns.set_palette('Set2')
prod_out_sorted = product_out.sort_values('price', ascending = True)
prod_out_hist = sns.barplot(data=prod_out_sorted, x = 'product_id', y = 'price')
plt.xticks(rotation=45)
prod_out_hist.set(xlabel='Identifiant Produit')
prod_out_hist.set(ylabel='Prix en euros')
plt.title("Produit valeurs aberrantes")
plt.grid(axis='y')

out_low = len(product_out[product_out['price'] < 100]) # -> 6
out_mid = len(product_out[(product_out['price'] >= 100) & (product_out['price'] < 150)]) # -> 20
out_high = len(product_out[product_out['price'] > 150]) # -> 6
print(out_low / len(product_out) * 100)
print(out_mid / len(product_out) * 100)
print(out_high / len(product_out) * 100)

#### 20 %  des outliers ont un prix entre 84 et 100 euros.

#### 60% des outliers ont un prix compris entre 100 et 150 euros.

#### 20 % des outliers ont un prix de plus de 150 euros.