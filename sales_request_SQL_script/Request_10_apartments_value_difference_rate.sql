WITH
	X AS
(SELECT
	ROUND(AVG(v.Valeur_fonciere/b.Surface_carrez), 2) AS Prix_m²_Appt_2p
FROM
	vente v,
    bien b
WHERE
	v.Id_bien = b.Id_bien
    AND b.Type_local = 'Appartement'
    AND b.Total_pieces = 2
GROUP BY
	b.Type_local),
    Y AS
    (SELECT
    ROUND(AVG(v.Valeur_fonciere/b.Surface_carrez), 2) AS Prix_m²_Appt_3p
FROM
	vente v,
    bien b
WHERE
	v.Id_bien = b.Id_bien
    AND b.Type_local = 'Appartement'
    AND b.Total_pieces = 3
GROUP BY
	b.Type_local)
SELECT
	*,
    ROUND(((Prix_m²_Appt_3p - Prix_m²_Appt_2p) / Prix_m²_Appt_2p) * 100,2) AS '%_Diff'
FROM X JOIN Y
    