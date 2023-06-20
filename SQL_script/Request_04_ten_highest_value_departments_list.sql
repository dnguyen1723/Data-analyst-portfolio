SELECT
	c.Code_departement,    
    ROUND(AVG(Valeur_fonciere/Surface_carrez),2) AS Prix_m²
FROM
	vente v,
    bien b,
    commune c
WHERE
	v.Id_bien = b.Id_bien
    AND c.Id_codedep_codecommune = b.Id_codedep_codecommune
GROUP BY c.Code_departement
ORDER BY Prix_m² DESC
LIMIT 10