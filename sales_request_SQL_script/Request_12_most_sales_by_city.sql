SELECT
	c.Nom_commune,
    c.population,
    COUNT(v.Id_vente) AS Nbre_transactions,
    ROUND(COUNT(v.Id_vente)/(Population/1000), 2) AS 'Transac/1kpers'
FROM
	vente v,
    bien b,
    commune c
WHERE
	v.Id_bien = b.Id_bien
    AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
    AND c.Population > 10000
GROUP BY
	c.Nom_commune, c.population
ORDER BY
	COUNT(v.Id_vente)/(Population/1000) DESC
LIMIT 20