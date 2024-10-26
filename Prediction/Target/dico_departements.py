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
    (87, 'Haute-Vienne'), (88, 'Vosges'), (89, 'Yonne'), (90, 'Territoire de Belfort'),
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
    '26-drome', '27-eure', '28-eure-et-loir', '29-finistere', '30-gard',
    #'2A-corse-du-sud', '2B-haute-corse',
    '30-gard', '31-haute-garonne',
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
    '95-val-d-oise',
    #'971-guadeloupe', '972-martinique', '973-guyane', '974-la-reunion', '976-mayotte'
]

# Départements déjà définis
SAISON_FEUX = {
    'departement-01-ain': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 10},
    'departement-25-doubs': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11},
    'departement-78-yvelines': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11},
    'departement-69-rhone': {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 10},
}

# Ajouter ou mettre à jour les départements restants
for dept in departements:
    dept_key = f'departement-{dept}'
    if dept_key not in SAISON_FEUX:
        SAISON_FEUX[dept_key] = {'jour_debut': 1, 'mois_debut': 3, 'jour_fin': 1, 'mois_fin': 11}