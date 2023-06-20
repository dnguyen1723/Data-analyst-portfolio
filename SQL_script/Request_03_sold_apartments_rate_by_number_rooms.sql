SELECT
	b.Type_local, 
    Total_pieces,
    ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER(),2) AS Proportion_ventes
FROM 
	vente v,
    bien b
WHERE 
	b.Id_bien = v.Id_vente
    AND b.Type_local = 'Appartement'
GROUP BY Total_pieces 
ORDER BY Total_pieces ;
