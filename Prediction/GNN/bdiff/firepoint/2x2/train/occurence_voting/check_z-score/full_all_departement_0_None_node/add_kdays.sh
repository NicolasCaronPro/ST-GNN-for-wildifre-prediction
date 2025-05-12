#!/bin/bash

# === VERIFICATION DES PARAMÈTRES ===
if [ -z "$1" ]; then
    echo "❌ Utilisation : $0 chemin/vers/dossier"
    exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Le chemin spécifié n'est pas un dossier valide : $TARGET_DIR"
    exit 1
fi

# === MOTIFS DE REMPLACEMENT ===
OLD_PART="_search_full_"
NEW_PART="_search_full_0_"

# === FICHIERS À RENOMMER ===
find "$TARGET_DIR" -type f -name "*$OLD_PART*" | while read -r path; do
    dir=$(dirname "$path")
    base=$(basename "$path")
    new_base=${base//$OLD_PART/$NEW_PART}
    new_path="$dir/$new_base"
    if [ "$path" != "$new_path" ]; then
        echo "📄 Renommage fichier : $path → $new_path"
        mv "$path" "$new_path"
    fi
done

# === DOSSIERS À RENOMMER (profonds d'abord) ===
find "$TARGET_DIR" -type d -name "*$OLD_PART*" | sort -r | while read -r path; do
    dir=$(dirname "$path")
    base=$(basename "$path")
    new_base=${base//$OLD_PART/$NEW_PART}
    new_path="$dir/$new_base"
    if [ "$path" != "$new_path" ]; then
        echo "📁 Renommage dossier : $path → $new_path"
        mv "$path" "$new_path"
    fi
done

echo "✅ Renommage terminé dans : $TARGET_DIR"
