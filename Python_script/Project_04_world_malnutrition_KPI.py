## 1 : Importation des librairies

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

## 2 : Importation des fichiers CSV

ss_nutri = pd.read_csv('sous_nutrition.csv')
dispo_alim = pd.read_csv('dispo_alimentaire.csv')
population = pd.read_csv('population.csv')
aide_alim = pd.read_csv('aide_alimentaire.csv')

## 3 : Aperçus des tables

ss_nutri.head()

dispo_alim.head()

population.head()

aide_alim.head()

## 4 : Vérification erreurs

### 4.1 : Table 'ss_nutri'

# Vérifier le type des variables -> OK
ss_nutri.dtypes

# Vérifier les valeurs manquantes
ss_nutri.isnull().sum() # 594 valeurs 'NaN' sur 1218 (49%) dans la colonne 'Valeur'

# Supprimer les lignes sans données dans la colonne 'Valeur'
ss_nutri = ss_nutri.dropna()

# Vérifier les doublons
ss_nutri.loc[ss_nutri[['Zone', 'Année', 'Valeur']].duplicated(keep=False), :] # Pas de doublons

# Remplacer valeurs de la colonne 'Année' : ex. '2012-2014' -> '2013'
ss_nutri['Année'].replace((['2012-2014', '2013-2015', '2014-2016', '2015-2017', '2016-2018','2017-2019']),['2013', '2014', '2015', '2016', '2017', '2018'], inplace=True)

# Remplacer la valeur '<0.1' de la colonne 'Valeur' par '0.05'
ss_nutri['Valeur'].replace({'<0.1':'0.05'}, inplace=True)

 ### 4.2 : table 'dispo_alim'

# Vérifier le type des variables -> OK
dispo_alim.dtypes

# Vérifier les valeurs manquantes -> erreurs, à modifier selon requêtes
dispo_alim.isnull().sum()

# Remplacement des valeurs manquantes de la colonne 'dispo alim kcal/pers/jour' par la moyenne de la colonne
dispo_alim.loc[dispo_alim['Disponibilité alimentaire (Kcal/personne/jour)'].isnull(), 'Disponibilité alimentaire (Kcal/personne/jour)'] = dispo_alim['Disponibilité alimentaire (Kcal/personne/jour)'].mean()

# Vérifier les doublons -> OK
duplicate_dispo_alim = dispo_alim[dispo_alim.duplicated()]
print(duplicate_dispo_alim)

### 4.3 : Table 'population'

# Vérifier le type des variables
population.dtypes # 'Année' int64
population['Année'] = population['Année'].astype(str) # Conversion de 'Année' en str

# Vérifier les valeurs manquantes -> OK
population.isnull().sum()

# Vérifier les doublons
population.loc[population[['Zone', 'Année', 'Valeur']].duplicated(keep=False), :] # Pas de doublons

### 4.4 : Table 'aide_alim'

# Vérifier le type des variables
aide_alim.dtypes # -> 'Année' int64
aide_alim['Année'] = aide_alim['Année'].astype(str) # Conversion de 'Année' en str

# Vérifier les valeurs manquantes -> OK
aide_alim.isnull().sum()

# Vérifier les doublons
aide_alim.loc[aide_alim[['Pays bénéficiaire', 'Année', 'Produit', 'Valeur']].duplicated(keep=False), :] # Pas de doublons

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## Requête Marc_1 : Proportion des personnes en état de sous-nutrition en 2017

# Fusion des tables 'ss_nutri' et 'population'
ss_nutri = pd.merge(ss_nutri, population, on = ['Zone', 'Année'], how = 'left')

# Renommer colonnes 'Valeur_x' et 'Valeur-y' par 'Sous-nutrition' et 'Population'
ss_nutri.rename(columns = {'Valeur_x':'Pop_ss_nutri'}, inplace = True)
ss_nutri.rename(columns = {'Valeur_y':'Population'}, inplace = True)

# Calcul pop 2017 en milliers de la table 'ss_nutri'
df_R1 = ss_nutri.loc[ss_nutri['Année'] == '2017', :]
sum_df_R1_2017 = round(df_R1['Population'].sum(), 2) # -> 4 144 279.2

# Calcul pop sous_nutrition 2017 en millions
ss_nutri_2017 = ss_nutri.loc[(ss_nutri['Année'] == '2017')]
sum_ssnutri_2017 = ss_nutri_2017['Pop_ss_nutri'].sum() # -> 535.7

# Calcul taux de sous-nutrition en 2017
tx_pop_ss_nutri = round(((sum_ssnutri_2017)/(sum_df_R1_2017))*100*1000, 2) # 12.88 %

### En 2017, 12.9 % des personnes sont en état de sous-nutrition.

## Requête Marc_2 : nombre de personnes pouvant être nourries en 2017

# Calcul pop totale 2017 en milliers
pop_2017 = population.loc[population['Année'] == '2017', :]
sum_pop_2017 = round(pop_2017['Valeur'].sum(), 3) # -> 7 548 134.111

# Fusion des tables 'dispo_alim' & 'pop_2017'
dispo_alim = pd.merge(dispo_alim, pop_2017, how='left', on='Zone')

# Renommer la colonne 'Valeur' par 'Population'
dispo_alim.rename(columns={'Valeur':'Population'}, inplace=True)

# Création de la variable 'Dispo alim pop jour'
dispo_alim['Dispo alim pop jour'] = dispo_alim['Population']*dispo_alim['Disponibilité alimentaire (Kcal/personne/jour)']

# Calcul de la population pouvant être nourries
total_pers_nourries = round(dispo_alim['Dispo alim pop jour'].sum()/2300*1000) # -> 7 864 136 274

# Calcul du taux de personnes pouvant être nourries
tx_pers_nourries = round((total_pers_nourries/sum_pop_2017)/1000, 2) # -> 1.04 %

### 7,9 milliards de personnes pourraient être nourries en 2017, soit 1.04 fois la population mondiale.


## Requête Marc_3 : nombre de personnes pouvant être nourries en 2017 à partir de la distribution alimentaire des produits végétaux

# Calcul de la dispo végétale pour la population par jour
dispo_veg_pop_jour = round(dispo_alim.loc[dispo_alim['Origine'] == 'vegetale', 'Dispo alim pop jour'].sum(), 2) # -> 15 468 839 586.86

# Calcul de la population pouvant être nourrie
Tot_pers_nourries_veg = round(dispo_veg_pop_jour / 2300, 2) # -> 6 725 582.43

# Calcul du taux de personnes pouvant être nourries par dispo_veg
tx_pop_veg = round(((dispo_veg_pop_jour/2300)/sum_pop_2017)*100, 2) # -> 89.1%

### 6,7 milliards de personnes pourraient être nourries à partir de la distribution alimentaire des produits végétaux en 2017, soit 89.1 % de la population mondiale.

## Requête Marc_4 : part de l'alimentation animale, des pertes et de l'alimentation humaine dans la disponibilité intérieure en 2017

# Calcul du total de la disponibilité intérieure
sum_dispo_int = dispo_alim['Disponibilité intérieure'].sum() # -> 9 848 994.0

# Calcul de la part de l'alimentation animale par rapport à la disponibilité intérieure
sum_alim_anim = dispo_alim['Aliments pour animaux'].sum() # -> 1 304 245.0
tx_alim_anim = round((sum_alim_anim/sum_dispo_int)*100, 2) # -> 13.24 %

# Calcul du taux de pertes par rapport à la disponibilité intérieure
sum_pertes = dispo_alim['Pertes'].sum() # -> 453698.0
tx_pertes = round((sum_pertes/sum_dispo_int)*100,2) # -> 4.61 %

# Calcul de la part de l'alimentation humaine par rapport à la disponibilité intérieure
sum_nourriture = dispo_alim['Nourriture'].sum() # -> 4876258.0
tx_nourriture = round((sum_nourriture/sum_dispo_int)*100,2) # -> 49.51 %

### En 2017, l'utilisation de la disponibilité intérieure est attribuée à :
### -  13.24 % pour l'alimentation animale
### -  4.61 % pour les pertes
### -  49.51 % pour l'alimentation humaine.

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## Requête Mélanie_1 : les pays dont la proportion de sous-nutrition est la plus élevée en 2017

# Fusion des tables 'pop_2017' & 'ss_nutri_2017'
pop_2017 = pd.merge(pop_2017, ss_nutri_2017, on='Zone', how='inner')
pop_2017.drop(columns='Année_x', inplace=True) # Suppression doublon colonne
pop_2017.rename(columns={'Année_y':'Année'}, inplace=True) # Renommer colonnes

# Création variable % de ss_nutri par rapport à la pop
pop_2017['tx_ss_nutri'] = round((pop_2017['Pop_ss_nutri']/pop_2017['Population'])*100000, 2)

# Trie décroissant des pays selon tx_ss_nutri
pop_2017.sort_values('tx_ss_nutri', ascending = False).head()

### Les Pays où le taux de sous-nutrition est le plus élevé en 2017 sont Haïti (48%), la Corée du Nord (47%) et Madagascar (41%).

## Requête Mélanie_2 : les pays ayant le plus bénéficié d'aides depuis 2013

# Calcul du total des aides perçues depuis 2013
sum_aide_alim = aide_alim.groupby(['Pays bénéficiaire']).sum()

# Trie par ordre décroissant
sum_aide_alim.sort_values('Valeur', ascending = False).head()

### Les pays ayant le plus bénéficié d'aides alimentaires depuis 2013 sont la Syrie (1 858 943 tonnes), l'Ethiopie (1 381 294 tonnes) et le Yémen (1 2064 484 tonnes).

## Requête Mélanie_3 : les pays ayant le plus de disponibilités alimentaires par habitant

#  Agrégation des données par pays
dispo_alim_pays = dispo_alim.groupby('Zone').sum()

# Trie par ordre décroissant
dispo_alim_pays.sort_values('Disponibilité alimentaire (Kcal/personne/jour)', ascending = False).head()

### Les pays ayant le plus de disponibilités alimentaires en Kcal/personne/an sont la Belgique (4154), le Luxembourg(4131) et la Turquie (4091).

## Requête Mélanie_4 : les pays ayant le moins de disponibilités alimentaires par habitant

# Trie par ordre croissant
dispo_alim_pays.sort_values('Disponibilité alimentaire (Kcal/personne/jour)', ascending = True).head()

### Les pays ayant le moins de disponibilités alimentaires en Kcal/personne/an sont la République centrafricaine (2053), le Zambie (2098) et la République démocratique de Corée (2163).

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## Requête Julien_1 : le lien logique entre les 10 variables du fichier de disponibilité alimentaire
### -> Nourriture + pertes + semences + traitement + aliments pour animaux + autres utilisations = **Disponibilité intérieure** = Production + importations - exportations + variations des stocks 

## Requête Julien_2 : répartition de l'utilisation des céréales entre l'alimentation humaine et l'alimentation animale

# Agrégation de la colonne 'Produit' du df 'Disponibilité alimentaire'
prod_anim_hum = dispo_alim.groupby('Produit')[['Aliments pour animaux', 'Nourriture', 'Disponibilité intérieure']].sum()
prod_anim_hum.reset_index(inplace=True)

# Création du df 'util_cereales'
util_cereales = prod_anim_hum.loc[(prod_anim_hum['Produit'] == 'Avoine') | (prod_anim_hum['Produit'] == 'Blé') | (prod_anim_hum['Produit'] == 'Céréales- Autres') | (prod_anim_hum['Produit'] == 'Maïs') | (prod_anim_hum['Produit'] == 'Millet') | (prod_anim_hum['Produit'] == 'Orge') | (prod_anim_hum['Produit'] == 'Seigle') | (prod_anim_hum['Produit'] == 'Sorgho') | (prod_anim_hum['Produit'] == 'Soja') | (prod_anim_hum['Produit'] == 'Riz (Eq Blanchi)'), :]
util_cereales.reset_index(inplace=True, drop=True)

# Calcul total utilisation céréales en tonnes
sum_dispo_cereale = util_cereales['Disponibilité intérieure'].sum() # -> 2 406 999.0

# # Part céréales pour l'alimentation humaine
tx_util_cereal_hum = round((util_cereales['Nourriture'].sum()/sum_dispo_cereale)*100, 2) # -> 42.75 %

# Part céréales pour les animaux
tx_util_cereal_anim = round((util_cereales['Aliments pour animaux'].sum()/sum_dispo_cereale)*100, 2) # -> 36.29 %

### 42.75 % des céréales sont destinés à l'alimentation humaine et 36.29 % à l'alimentation des animaux.

## Requête Julien_3 : Thaïlande, part d'exportation/production du manioc et % personnes en sous-nutrition

# Exportations et Production du manioc
thai_manioc_exp_prod = dispo_alim.loc[(dispo_alim['Zone'] == 'Thaïlande') & (dispo_alim['Produit'] == 'Manioc'), ['Zone', 'Produit', 'Exportations - Quantité', 'Production']]

# Total production manioc en milliers de tonnes
sum_prod_manioc = thai_manioc_exp_prod['Production'].sum() # -> 30 228.0

# Total exportation manioc en milliers de tonnes
sum_exp_manioc = thai_manioc_exp_prod['Exportations - Quantité'].sum() # -> 25 214.0

# Taux d'exportation maniox par rapport à la production
tx_exp_prod_manioc = round((sum_exp_manioc / sum_prod_manioc) * 100, 2) # -> 83.41 %

# Calcul pop totale 2018 en milliers
pop_2018 = population.loc[population['Année'] == '2018', :]

# Calcul sous_nutri 2018 en millions
ss_nutri_2018 = ss_nutri.loc[(ss_nutri['Année'] == '2018')]

# Fusion des tables 'pop_2018' & 'ss_nutri_2018'
pop_2018 = pd.merge(pop_2018, ss_nutri_2018, on='Zone', how='inner')
pop_2018.drop(columns='Année_x', inplace=True) # Suppression doublon colonne
pop_2018.rename(columns={'Valeur_x':'Population','Année_y':'Année','Valeur_y':'pop_ss_nutri'}, inplace=True) # Renommer colonnes

# Filtrage données Thaïlande
thai_pop_2018 = pop_2018.loc[pop_2018['Zone'] == 'Thaïlande']

# Total pop thaï 2018 milliers
sum_pop_thai_2018 = round(thai_pop_2018['Population'].sum(), 2) # -> 69 428.45

# Total sous_nutri thaï 2018 en millions
sum_ss_nutri_thai_2018 = thai_pop_2018['Pop_ss_nutri'].sum() # -> 6.5

# Taux sous_nutri/population
tx_ss_nutri_thai_2018 = round(((sum_ss_nutri_thai_2018 / sum_pop_thai_2018) * 100) * 1000, 2) # -> 9.36 %

### En 2018 en Thaïlande, le taux de personnes en sous-nutrition est de 9.36 % et le taux d'exportation du manioc par rapport à la production est de 83.41 %.