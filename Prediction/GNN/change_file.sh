#!/bin/bash

# Vérifier qu'un répertoire a été fourni en argument
if [ -z "$1" ]; then
    echo "Usage: $0 <répertoire>"
    exit 1
fi

# Première passe : renommer les fichiers et dossiers contenant *_search_one_*
#find "$1" -depth -name "*_search_one_*" | while IFS= read -r path; do
#    new_path=$(echo "$path" | sed 's/_search_one_/_search_full_one_/g')
#    mv "$path" "$new_path"
#    echo "Renommé : $path -> $new_path"
#done

# Deuxième passe : renommer les fichiers et dossiers contenant *_full_one_*
#find "$1" -depth -name "*_full_one_*" | while IFS= read -r path; do
#    # Vérifier que le chemin ne contient pas déjà _full_full_one_
#    if [[ "$path" == *_full_full_one_* ]]; then
#        continue
#    fi
#    new_path=$(echo "$path" | sed 's/_full_one_/_full_full_one_/g')
#    mv "$path" "$new_path"
#    echo "Renommé : $path -> $new_path"
#done

# Troisième passe : modifier les fichiers et dossiers de la forme *_search_full_full_* en *_search_full_
#find "$1" -depth -name "*_search_full_full_*" | while IFS= read -r path; do
#    new_path=$(echo "$path" | sed 's/_search_full_full_/_search_full_/g')
#    mv "$path" "$new_path"
#    echo "Renommé : $path -> $new_path"
#done

# Quatrième passe : modifier les fichiers et dossiers de la forme filter*_full_full_ en filter*_search_full_
#find "$1" -depth -name "filter*_full_full_*" | while IFS= read -r path; do
#    new_path=$(echo "$path" | sed 's/_full_full_/_search_full_/g')
#    mv "$path" "$new_path"
#    echo "Renommé : $path -> $new_path"
#done

find "$1" -depth -name "*_full_one_*" | while IFS= read -r path; do
    new_path=$(echo "$path" | sed 's/_full_one_/_full_all_one_/g')
    mv "$path" "$new_path"
    echo "Renommé : $path -> $new_path"
done