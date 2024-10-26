#!/bin/bash

# Activer l'option nullglob pour que les motifs non correspondants se résolvent en une liste vide
shopt -s nullglob

# Chemin de base où se trouvent les fichiers ZIP
source_folder="/home/caron/Téléchargements"  # Remplacez par le chemin réel où se trouvent les fichiers ZIP

# Chemin de destination de base où les fichiers doivent être déplacés
destination_base="/media/caron/X9 Pro/travaille/Thèse/csv"

# Liste des départements
declare -A departements=(
    ["01"]="ain"
    ["02"]="aisne"
    ["03"]="allier"
    ["04"]="alpes-de-haute-provence"
    ["05"]="hautes-alpes"
    ["06"]="alpes-maritimes"
    ["07"]="ardeche"
    ["08"]="ardennes"
    ["09"]="ariege"
    ["10"]="aube"
    ["11"]="aude"
    ["12"]="aveyron"
    ["13"]="bouches-du-rhone"
    ["14"]="calvados"
    ["15"]="cantal"
    ["16"]="charente"
    ["17"]="charente-maritime"
    ["18"]="cher"
    ["19"]="correze"
    ["21"]="cote-d-or"
    ["22"]="cotes-d-armor"
    ["23"]="creuse"
    ["24"]="dordogne"
    ["25"]="doubs"
    ["26"]="drome"
    ["27"]="eure"
    ["28"]="eure-et-loir"
    ["29"]="finistere"
    ["30"]="gard"
    ["31"]="haute-garonne"
    ["32"]="gers"
    ["33"]="gironde"
    ["34"]="herault"
    ["36"]="indre"
    ["37"]="indre-et-loire"
    ["38"]="isere"
    ["39"]="jura"
    ["40"]="landes"
    ["41"]="loir-et-cher"
    ["42"]="loire"
    ["43"]="haute-loire"
    ["44"]="loire-atlantique"
    ["45"]="loiret"
    ["46"]="lot"
    ["47"]="lot-et-garonne"
    ["48"]="lozere"
    ["49"]="maine-et-loire"
    ["50"]="manche"
    ["51"]="marne"
    ["52"]="haute-marne"
    ["53"]="mayenne"
    ["54"]="meurthe-et-moselle"
    ["55"]="meuse"
    ["56"]="morbihan"
    ["57"]="moselle"
    ["58"]="nievre"
    ["59"]="nord"
    ["60"]="oise"
    ["61"]="orne"
    ["62"]="pas-de-calais"
    ["63"]="puy-de-dome"
    ["64"]="pyrenees-atlantiques"
    ["65"]="hautes-pyrenees"
    ["66"]="pyrenees-orientales"
    ["67"]="bas-rhin"
    ["68"]="haut-rhin"
    ["69"]="rhone"
    ["70"]="haute-saone"
    ["71"]="saone-et-loire"
    ["72"]="sarthe"
    ["73"]="savoie"
    ["74"]="haute-savoie"
    ["75"]="paris"
    ["76"]="seine-maritime"
    ["77"]="seine-et-marne"
    ["78"]="yvelines"
    ["79"]="deux-sevres"
    ["80"]="somme"
    ["81"]="tarn"
    ["82"]="tarn-et-garonne"
    ["83"]="var"
    ["84"]="vaucluse"
    ["85"]="vendee"
    ["86"]="vienne"
    ["87"]="haute-vienne"
    ["88"]="vosges"
    ["89"]="yonne"
    ["90"]="territoire-de-belfort"
    ["91"]="essonne"
    ["92"]="hauts-de-seine"
    ["93"]="seine-saint-denis"
    ["94"]="val-de-marne"
)

# Vérifier si le dossier source existe
if [ ! -d "$source_folder" ]; then
    echo "Le dossier source '$source_folder' n'existe pas."
    exit 1
fi

# Vérifier si le dossier de destination existe, sinon le créer
if [ ! -d "$destination_base" ]; then
    mkdir -p "$destination_base"
fi

# Tableau des fichiers ZIP correspondants
files=("$source_folder"/CoSIA_D0*.zip)

# Vérifier s'il y a des fichiers à traiter
if [ ${#files[@]} -eq 0 ]; then
    echo "Aucun fichier correspondant trouvé dans '$source_folder'."
    exit 0
fi

# Boucle sur tous les fichiers ZIP qui correspondent au modèle "CoSIA_D0{dept}_*.zip"
for file in "${files[@]}"; do
    # Extraire le nom de fichier sans le chemin
    filename=$(basename "$file")
    
    # Extraire le code du département à partir du nom de fichier
    dept=$(echo "$filename" | sed -n 's/^CoSIA_D0\([0-9]\{2\}\)_.*\.zip$/\1/p')
    
    # Vérifier si le code département est valide
    if [[ -n "$dept" && -n "${departements[$dept]}" ]]; then
        dept_name="departement-$dept-${departements[$dept]}"
        
        # Chemin du dossier de destination pour le département
        destination_folder="$destination_base/$dept_name/data/cosia"
        mkdir -p "$destination_folder"
        
        # Déplacer le fichier dans le bon dossier
        mv "$file" "$destination_folder"
        
        echo "Le fichier '$filename' a été déplacé vers '$destination_folder'."
    else
        echo "Aucun département valide trouvé pour le fichier : '$filename'."
    fi
done
