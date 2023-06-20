SELECT
	Nom_region, COUNT(*) AS Nombre_ventes
FROM 
	vente v,
    bien b,
    commune c,
    region r
WHERE 
	b.Id_bien = v.Id_bien
    AND b.Type_local = 'Appartement'
    AND c.Id_region = r.Id_region
    AND c.Id_codedep_codecommune = b.Id_codedep_codecommune
GROUP BY Nom_region
ORDER BY Nombre_ventes DESC ;