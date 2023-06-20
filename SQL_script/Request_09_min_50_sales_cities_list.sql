SELECT
	c.Nom_commune,
    COUNT(v.Id_vente) AS Nbre_ventes_trim1
FROM
	vente v,
    bien b,
    commune c
WHERE
	v.Id_bien = b.Id_bien
    AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
    AND QUARTER(v.Date_mutation) = 1
GROUP BY
	c.Nom_commune
HAVING
	COUNT(v.Id_vente) >= 50
ORDER BY
	Nbre_ventes_trim1 DESC
    