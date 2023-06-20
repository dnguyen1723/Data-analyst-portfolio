SELECT
	*
FROM
(SELECT
	*,
   	RANK() OVER (PARTITION BY R.Code_departement ORDER BY R.Valeur_fonciere DESC) AS Classement_Regional
FROM
(SELECT
	c.Nom_commune,
    c.Code_departement,
    ROUND(AVG(v.valeur_fonciere), 2) AS Valeur_fonciere
FROM
	vente v,
    bien b,
    commune c
WHERE
	v.Id_bien = b.Id_bien
    AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
    AND c.Code_departement IN (6, 13, 33, 59, 69)
GROUP BY
	c.Nom_commune, c.Code_departement
ORDER BY
	Code_departement DESC) AS R) AS S
WHERE
	Classement_regional <= 3