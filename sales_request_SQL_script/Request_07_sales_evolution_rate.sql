WITH
	X AS (SELECT
	Count(v.Id_vente) AS Nbre_ventes_trim1
FROM
	vente v
WHERE
	v.Date_mutation BETWEEN '2020-01-01' AND '2020-03-31'),
    Y AS (SELECT
	Count(v.Id_vente) AS Nbre_ventes_trim2
FROM
	vente v
WHERE
	v.Date_mutation BETWEEN '2020-04-01' AND '2020-06-30')
SELECT
	*,
	round(((Nbre_ventes_trim2 - Nbre_ventes_trim1) / Nbre_ventes_trim1) * 100,2) AS Taux_evolution
FROM X JOIN Y ;