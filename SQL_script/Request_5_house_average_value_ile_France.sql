SELECT
	b.Type_local,
    r.Nom_region,
    ROUND(AVG(v.Valeur_fonciere/Surface_carrez), 2) AS Prix_moyen_m²    
FROM
	vente v,
    bien b,
    commune c,
    region r
WHERE
	v.Id_bien = b.Id_bien
    AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
    AND c.Id_region = r.Id_region
    AND r.Nom_region = 'Ile-de-France'
    AND b.Type_local = 'Maison'