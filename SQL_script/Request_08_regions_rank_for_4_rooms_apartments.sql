SELECT
	DENSE_RANK() OVER (ORDER BY (AVG(v.Valeur_fonciere/Surface_carrez))DESC) Classement,
	r.Nom_region,
    ROUND(AVG(v.Valeur_fonciere/Surface_carrez), 2) AS 'Prix_m²_appart5p+'
FROM
	vente v,
    bien b,
    commune c,
    region r
WHERE
		v.Id_bien = b.Id_bien
        AND b.Id_codedep_codecommune = c.Id_codedep_codecommune
        AND c.Id_region = r.Id_region
        AND b.Type_local = 'Appartement'
        AND b.Total_pieces > 4
GROUP BY
	Nom_region
ORDER BY 'Prix_m²_appart5p+' DESC