SELECT
	b.Type_local, COUNT(*) AS Nombre_total_ventes
FROM 
	vente v,
    bien b
WHERE 
	b.Id_bien = v.Id_bien
    AND b.Type_local = 'Appartement' ;