SELECT
	v.Id_bien,
    r.Nom_region,
    c.Code_departement,
    b.Surface_carrez,
    v.Valeur_fonciere    
FROM
	vente v,
    bien b,
    commune c,
    region r
WHERE
	v.Id_bien = b.Id_bien
    AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
    AND r.Id_region = c.Id_region
    AND b.Type_local = 'Appartement'
ORDER BY v.Valeur_fonciere DESC
LIMIT 10