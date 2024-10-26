# Liste des départements avec leurs codes
departements = [
    (1, 'Ain'), (2, 'Aisne'), (3, 'Allier'), (4, 'Alpes-de-Haute-Provence'),
    (5, 'Hautes-Alpes'), (6, 'Alpes-Maritimes'), (7, 'Ardeche'), (8, 'Ardennes'),
    (9, 'Ariege'), (10, 'Aube'), (11, 'Aude'), (12, 'Aveyron'),
    (13, 'Bouches-du-Rhone'), (14, 'Calvados'), (15, 'Cantal'), (16, 'Charente'),
    (17, 'Charente-Maritime'), (18, 'Cher'), (19, 'Correze'), (21, 'Cote-d-Or'),
    (22, 'Cotes-d-Armor'), (23, 'Creuse'), (24, 'Dordogne'), (25, 'Doubs'),
    (26, 'Drome'), (27, 'Eure'), (28, 'Eure-et-Loir'), (29, 'Finistere'),
    ('2A', 'Corse-du-Sud'), ('2B', 'Haute-Corse'), (30, 'Gard'), (31, 'Haute-Garonne'),
    (32, 'Gers'), (33, 'Gironde'), (34, 'Herault'), (35, 'Ille-et-Vilaine'),
    (36, 'Indre'), (37, 'Indre-et-Loire'), (38, 'Isere'), (39, 'Jura'),
    (40, 'Landes'), (41, 'Loir-et-Cher'), (42, 'Loire'), (43, 'Haute-Loire'),
    (44, 'Loire-Atlantique'), (45, 'Loiret'), (46, 'Lot'), (47, 'Lot-et-Garonne'),
    (48, 'Lozere'), (49, 'Maine-et-Loire'), (50, 'Manche'), (51, 'Marne'),
    (52, 'Haute-Marne'), (53, 'Mayenne'), (54, 'Meurthe-et-Moselle'), (55, 'Meuse'),
    (56, 'Morbihan'), (57, 'Moselle'), (58, 'Nievre'), (59, 'Nord'),
    (60, 'Oise'), (61, 'Orne'), (62, 'Pas-de-Calais'), (63, 'Puy-de-Dome'),
    (64, 'Pyrenees-Atlantiques'), (65, 'Hautes-Pyrenees'), (66, 'Pyrenees-Orientales'),
    (67, 'Bas-Rhin'), (68, 'Haut-Rhin'), (69, 'Rhone'), (70, 'Haute-Saone'),
    (71, 'Saone-et-Loire'), (72, 'Sarthe'), (73, 'Savoie'), (74, 'Haute-Savoie'),
    (75, 'Paris'), (76, 'Seine-Maritime'), (77, 'Seine-et-Marne'), (78, 'Yvelines'),
    (79, 'Deux-Sevres'), (80, 'Somme'), (81, 'Tarn'), (82, 'Tarn-et-Garonne'),
    (83, 'Var'), (84, 'Vaucluse'), (85, 'Vendee'), (86, 'Vienne'),
    (87, 'Haute-Vienne'), (88, 'Vosges'), (89, 'Yonne'), (90, 'Territoire-de-Belfort'),
    (91, 'Essonne'), (92, 'Hauts-de-Seine'), (93, 'Seine-Saint-Denis'), (94, 'Val-de-Marne'),
    (95, 'Val-d-Oise'), (971, 'Guadeloupe'), (972, 'Martinique'), (973, 'Guyane'),
    (974, 'La Reunion'), (976, 'Mayotte')
]

int2str = {code: name.lower().replace("'", "-") for code, name in departements}
int2strMaj = {code: name for code, name in departements}
int2name = {code: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements}

str2int = {name.lower().replace("'", "-"): code for code, name in departements}
str2intMaj = {name: code for code, name in departements}
str2name = {name: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements}

name2str = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name for code, name in departements}
name2int = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": code for code, name in departements}
name2strlow = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name.lower() for code, name in departements}
name2intstr = {
    f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": (f'0{code}' if code not in ['2A', '2B'] and int(code) < 10 else str(code))
    for code, name in departements
}

departements = [
    '01-ain', '02-aisne', '03-allier', '04-alpes-de-haute-provence',
    '05-hautes-alpes', '06-alpes-maritimes', '07-ardeche', '08-ardennes',
    '09-ariege', '10-aube', '11-aude', '12-aveyron',
    '13-bouches-du-rhone', '14-calvados', '15-cantal', '16-charente',
    '17-charente-maritime', '18-cher', '19-correze', '21-cote-d-or',
    '22-cotes-d-armor', '23-creuse', '24-dordogne', '25-doubs',
    '26-drome', '27-eure', '28-eure-et-loir', '29-finistere',
    '2A-corse-du-sud', '2B-haute-corse', '30-gard', '31-haute-garonne',
    '32-gers', '33-gironde', '34-herault', '35-ille-et-vilaine',
    '36-indre', '37-indre-et-loire', '38-isere', '39-jura',
    '40-landes', '41-loir-et-cher', '42-loire', '43-haute-loire',
    '44-loire-atlantique', '45-loiret', '46-lot', '47-lot-et-garonne',
    '48-lozere', '49-maine-et-loire', '50-manche', '51-marne',
    '52-haute-marne', '53-mayenne', '54-meurthe-et-moselle', '55-meuse',
    '56-morbihan', '57-moselle', '58-nievre', '59-nord',
    '60-oise', '61-orne', '62-pas-de-calais', '63-puy-de-dome',
    '64-pyrenees-atlantiques', '65-hautes-pyrenees', '66-pyrenees-orientales',
    '67-bas-rhin', '68-haut-rhin', '69-rhone', '70-haute-saone',
    '71-saone-et-loire', '72-sarthe', '73-savoie', '74-haute-savoie',
    '75-paris', '76-seine-maritime', '77-seine-et-marne', '78-yvelines',
    '79-deux-sevres', '80-somme', '81-tarn', '82-tarn-et-garonne',
    '83-var', '84-vaucluse', '85-vendee', '86-vienne',
    '87-haute-vienne', '88-vosges', '89-yonne', '90-territoire-de-belfort',
    '91-essonne', '92-hauts-de-seine', '93-seine-saint-denis', '94-val-de-marne',
    '95-val-d-oise', '971-guadeloupe', '972-martinique', '973-guyane',
    '974-la-reunion', '976-mayotte'
]

# Départements déjà définis
SAISON_FEUX = {
    'departement-01-ain': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 10},
    'departement-25-doubs': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11},
    'departement-78-yvelines': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11},
    'departement-69-rhone': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 10},
}

dico_foret_url = {
    "departement-01-ain": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D001_2014-04-01/BDFORET_2-0__SHP_LAMB93_D001_2014-04-01.7z",
    "departement-02-aisne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D002_2017-10-04/BDFORET_2-0__SHP_LAMB93_D002_2017-10-04.7z",
    "departement-03-allier": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D003_2014-04-01/BDFORET_2-0__SHP_LAMB93_D003_2014-04-01.7z",
    "departement-04-alpes-de-haute-provence": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D004_2014-04-01/BDFORET_2-0__SHP_LAMB93_D004_2014-04-01.7z",
    "departement-05-hautes-alpes": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D005_2015-04-01/BDFORET_2-0__SHP_LAMB93_D005_2015-04-01.7z",
    "departement-06-alpes-maritimes": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D006_2021-03-01/BDFORET_2-0__SHP_LAMB93_D006_2021-03-01.7z",
    "departement-07-ardeche": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D007_2014-04-01/BDFORET_2-0__SHP_LAMB93_D007_2014-04-01.7z",
    "departement-08-ardennes": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D008_2022-04-01/BDFORET_2-0__SHP_LAMB93_D008_2022-04-01.7z",
    "departement-09-ariege": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D009_2017-10-04/BDFORET_2-0__SHP_LAMB93_D009_2017-10-04.7z",
    "departement-10-aube": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D010_2021-03-01/BDFORET_2-0__SHP_LAMB93_D010_2021-03-01.7z",
    "departement-11-aude": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D011_2018-11-20/BDFORET_2-0__SHP_LAMB93_D011_2018-11-20.7z",
    "departement-12-aveyron": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D012_2014-04-01/BDFORET_2-0__SHP_LAMB93_D012_2014-04-01.7z",
    "departement-13-bouches-du-rhone": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D013_2014-04-01/BDFORET_2-0__SHP_LAMB93_D013_2014-04-01.7z",
    "departement-14-calvados": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D014_2014-04-01/BDFORET_2-0__SHP_LAMB93_D014_2014-04-01.7z",
    "departement-15-cantal": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D015_2015-04-01/BDFORET_2-0__SHP_LAMB93_D015_2015-04-01.7z",
    "departement-16-charente": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D016_2018-01-15/BDFORET_2-0__SHP_LAMB93_D016_2018-01-15.7z",
    "departement-17-charente-maritime": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D017_2017-10-04/BDFORET_2-0__SHP_LAMB93_D017_2017-10-04.7z",
    "departement-18-cher": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D018_2014-04-01/BDFORET_2-0__SHP_LAMB93_D018_2014-04-01.7z",
    "departement-19-correze": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D019_2016-02-16/BDFORET_2-0__SHP_LAMB93_D019_2016-02-16.7z",
    "departement-2A-corse-du-sud": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D02A_2017-05-10/BDFORET_2-0__SHP_LAMB93_D02A_2017-05-10.7z",
    "departement-2B-haute-corse": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D02B_2016-02-16/BDFORET_2-0__SHP_LAMB93_D02B_2016-02-16.7z",
    "departement-21-cote-d-or": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D021_2017-10-04/BDFORET_2-0__SHP_LAMB93_D021_2017-10-04.7z",
    "departement-22-cotes-d-armor": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D022_2018-10-01/BDFORET_2-0__SHP_LAMB93_D022_2018-10-01.7z",
    "departement-23-creuse": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D023_2017-05-10/BDFORET_2-0__SHP_LAMB93_D023_2017-05-10.7z",
    "departement-24-dordogne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D024_2019-01-09/BDFORET_2-0__SHP_LAMB93_D024_2019-01-09.7z",
    "departement-25-doubs": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D025_2016-02-16/BDFORET_2-0__SHP_LAMB93_D025_2016-02-16.7z",
    "departement-26-drome": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D026_2014-04-01/BDFORET_2-0__SHP_LAMB93_D026_2014-04-01.7z",
    "departement-27-eure": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D027_2014-04-01/BDFORET_2-0__SHP_LAMB93_D027_2014-04-01.7z",
    "departement-28-eure-et-loir": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D028_2017-05-10/BDFORET_2-0__SHP_LAMB93_D028_2017-05-10.7z",
    "departement-29-finistere": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D029_2014-04-01/BDFORET_2-0__SHP_LAMB93_D029_2014-04-01.7z",
    "departement-30-gard": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D030_2018-11-20/BDFORET_2-0__SHP_LAMB93_D030_2018-11-20.7z",
    "departement-31-haute-garonne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D031_2019-01-09/BDFORET_2-0__SHP_LAMB93_D031_2019-01-09.7z",
    "departement-32-gers": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D032_2015-04-01/BDFORET_2-0__SHP_LAMB93_D032_2015-04-01.7z",
    "departement-33-gironde": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D033_2017-05-10/BDFORET_2-0__SHP_LAMB93_D033_2017-05-10.7z",
    "departement-34-herault": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D034_2018-01-15/BDFORET_2-0__SHP_LAMB93_D034_2018-01-15.7z",
    "departement-35-ille-et-vilaine": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D035_2018-10-01/BDFORET_2-0__SHP_LAMB93_D035_2018-10-01.7z",
    "departement-36-indre": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D036_2021-03-01/BDFORET_2-0__SHP_LAMB93_D036_2021-03-01.7z",
    "departement-37-indre-et-loire": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D037_2017-05-10/BDFORET_2-0__SHP_LAMB93_D037_2017-05-10.7z",
    "departement-38-isere": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D038_2014-04-01/BDFORET_2-0__SHP_LAMB93_D038_2014-04-01.7z",
    "departement-39-jura": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D039_2016-02-16/BDFORET_2-0__SHP_LAMB93_D039_2016-02-16.7z",
    "departement-40-landes": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D040_2018-10-01/BDFORET_2-0__SHP_LAMB93_D040_2018-10-01.7z",
    "departement-41-loir-et-cher": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D041_2017-05-10/BDFORET_2-0__SHP_LAMB93_D041_2017-05-10.7z",
    "departement-42-loire": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D042_2014-04-01/BDFORET_2-0__SHP_LAMB93_D042_2014-04-01.7z",
    "departement-43-haute-loire": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D043_2021-11-15/BDFORET_2-0__SHP_LAMB93_D043_2021-11-15.7z",
    "departement-44-loire-atlantique": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D044_2018-10-01/BDFORET_2-0__SHP_LAMB93_D044_2018-10-01.7z",
    "departement-45-loiret": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D045_2015-04-01/BDFORET_2-0__SHP_LAMB93_D045_2015-04-01.7z",
    "departement-46-lot": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D046_2017-05-10/BDFORET_2-0__SHP_LAMB93_D046_2017-05-10.7z",
    "departement-47-lot-et-garonne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D047_2018-10-01/BDFORET_2-0__SHP_LAMB93_D047_2018-10-01.7z",
    "departement-48-lozere": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D048_2017-05-10/BDFORET_2-0__SHP_LAMB93_D048_2017-05-10.7z",
    "departement-49-maine-et-loire": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D049_2018-10-01/BDFORET_2-0__SHP_LAMB93_D049_2018-10-01.7z",
    "departement-50-manche": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D050_2018-10-01/BDFORET_2-0__SHP_LAMB93_D050_2018-10-01.7z",
    "departement-51-marne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D051_2021-03-01/BDFORET_2-0__SHP_LAMB93_D051_2021-03-01.7z",
    "departement-52-haute-marne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D052_2014-04-01/BDFORET_2-0__SHP_LAMB93_D052_2014-04-01.7z",
    "departement-53-mayenne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D053_2018-10-01/BDFORET_2-0__SHP_LAMB93_D053_2018-10-01.7z",
    "departement-54-meurthe-et-moselle": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D054_2014-04-01/BDFORET_2-0__SHP_LAMB93_D054_2014-04-01.7z",
    "departement-55-meuse": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D055_2015-04-01/BDFORET_2-0__SHP_LAMB93_D055_2015-04-01.7z",
    "departement-56-morbihan": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D056_2021-03-01/BDFORET_2-0__SHP_LAMB93_D056_2021-03-01.7z",
    "departement-57-moselle": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D057_2014-04-01/BDFORET_2-0__SHP_LAMB93_D057_2014-04-01.7z",
    "departement-58-nievre": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D058_2014-04-01/BDFORET_2-0__SHP_LAMB93_D058_2014-04-01.7z",
    "departement-59-nord": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D059_2014-04-01/BDFORET_2-0__SHP_LAMB93_D059_2014-04-01.7z",
    "departement-60-oise": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D060_2015-04-01/BDFORET_2-0__SHP_LAMB93_D060_2015-04-01.7z",
    "departement-61-orne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D061_2014-04-01/BDFORET_2-0__SHP_LAMB93_D061_2014-04-01.7z",
    "departement-62-pas-de-calais": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D062_2014-04-01/BDFORET_2-0__SHP_LAMB93_D062_2014-04-01.7z",
    "departement-63-puy-de-dome": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D063_2014-04-01/BDFORET_2-0__SHP_LAMB93_D063_2014-04-01.7z",
    "departement-64-pyrenees-atlantiques": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D064_2014-04-01/BDFORET_2-0__SHP_LAMB93_D064_2014-04-01.7z",
    "departement-65-hautes-pyrenees": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D065_2014-04-01/BDFORET_2-0__SHP_LAMB93_D065_2014-04-01.7z",
    "departement-66-pyrenees-orientales": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D066_2018-11-20/BDFORET_2-0__SHP_LAMB93_D066_2018-11-20.7z",
    "departement-67-bas-rhin": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D067_2014-04-01/BDFORET_2-0__SHP_LAMB93_D067_2014-04-01.7z",
    "departement-68-haut-rhin": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D068_2014-04-01/BDFORET_2-0__SHP_LAMB93_D068_2014-04-01.7z",
    "departement-69-rhone": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D069_2014-04-01/BDFORET_2-0__SHP_LAMB93_D069_2014-04-01.7z",
    "departement-70-haute-saone": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D070_2015-06-01/BDFORET_2-0__SHP_LAMB93_D070_2015-06-01.7z",
    "departement-71-saone-et-loire": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D071_2018-11-20/BDFORET_2-0__SHP_LAMB93_D071_2018-11-20.7z",
    "departement-72-sarthe": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D072_2022-04-01/BDFORET_2-0__SHP_LAMB93_D072_2022-04-01.7z",
    "departement-73-savoie": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D073_2014-04-01/BDFORET_2-0__SHP_LAMB93_D073_2014-04-01.7z",
    "departement-74-haute-savoie": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D074_2014-04-01/BDFORET_2-0__SHP_LAMB93_D074_2014-04-01.7z",
    "departement-75-paris": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D075_2018-01-15/BDFORET_2-0__SHP_LAMB93_D075_2018-01-15.7z",
    "departement-76-seine-maritime": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D076_2015-06-01/BDFORET_2-0__SHP_LAMB93_D076_2015-06-01.7z",
    "departement-77-seine-et-marne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D077_2017-10-04/BDFORET_2-0__SHP_LAMB93_D077_2017-10-04.7z",
    "departement-78-yvelines": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D078_2019-01-09/BDFORET_2-0__SHP_LAMB93_D078_2019-01-09.7z",
    "departement-79-deux-sevres": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D079_2014-04-01/BDFORET_2-0__SHP_LAMB93_D079_2014-04-01.7z",
    "departement-80-somme": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D080_2015-04-01/BDFORET_2-0__SHP_LAMB93_D080_2015-04-01.7z",
    "departement-81-tarn": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D081_2014-04-01/BDFORET_2-0__SHP_LAMB93_D081_2014-04-01.7z",
    "departement-82-tarn-et-garonne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D082_2017-10-04/BDFORET_2-0__SHP_LAMB93_D082_2017-10-04.7z",
    "departement-83-var": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D083_2015-04-01/BDFORET_2-0__SHP_LAMB93_D083_2015-04-01.7z",
    "departement-84-vaucluse": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D084_2022-04-01/BDFORET_2-0__SHP_LAMB93_D084_2022-04-01.7z",
    "departement-85-vendee": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D085_2014-04-01/BDFORET_2-0__SHP_LAMB93_D085_2014-04-01.7z",
    "departement-86-vienne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D086_2014-04-01/BDFORET_2-0__SHP_LAMB93_D086_2014-04-01.7z",
    "departement-87-haute-vienne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D087_2016-02-16/BDFORET_2-0__SHP_LAMB93_D087_2016-02-16.7z",
    "departement-88-vosges": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D088_2015-04-01/BDFORET_2-0__SHP_LAMB93_D088_2015-04-01.7z",
    "departement-89-yonne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D089_2014-04-01/BDFORET_2-0__SHP_LAMB93_D089_2014-04-01.7z",
    "departement-90-territoire-de-belfort": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D090_2015-06-01/BDFORET_2-0__SHP_LAMB93_D090_2015-06-01.7z",
    "departement-91-essonne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D091_2018-01-15/BDFORET_2-0__SHP_LAMB93_D091_2018-01-15.7z",
    "departement-92-hauts-de-seine": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D092_2018-01-15/BDFORET_2-0__SHP_LAMB93_D092_2018-01-15.7z",
    "departement-93-seine-saint-denis": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D093_2018-01-15/BDFORET_2-0__SHP_LAMB93_D093_2018-01-15.7z",
    "departement-94-val-de-marne": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D094_2017-10-04/BDFORET_2-0__SHP_LAMB93_D094_2017-10-04.7z",
    "departement-95-val-d-oise": "https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D095_2015-04-01/BDFORET_2-0__SHP_LAMB93_D095_2015-04-01.7z"
}


# Ajouter ou mettre à jour les départements restants
for dept in departements:
    dept_key = f'departement-{dept}'
    if dept_key not in SAISON_FEUX:
        SAISON_FEUX[dept_key] = {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11}