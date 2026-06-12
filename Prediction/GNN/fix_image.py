import os
from PIL import Image, ImageDraw, ImageFont

img_path = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_01_06_25/check_z-score/full_all_3_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node/NetMLP_search_full_0_10_all_one_nbsinister_regression_ccllt-id{node}-nclusters{30}/runs_variance_evolution.png'
img = Image.open(img_path)

# Extrait les 4 graphiques souhaités en rognant l'image originale (1800x1500)
agg = img.crop((0, 0, 900, 500))
k_comps = img.crop((0, 500, 900, 1000))
recall = img.crop((900, 500, 1800, 1000))
spline = img.crop((0, 1000, 1800, 1500))

# Nouvelle image avec fond blanc
new_img = Image.new('RGB', (1800, 1500), 'white')

# Nouvelle disposition (supprime le graphique "Macro scores" en haut à droite)
new_img.paste(agg, (0, 0))            # Haut gauche
new_img.paste(k_comps, (900, 0))      # Haut droite
new_img.paste(recall, (450, 500))     # Milieu centre
new_img.paste(spline, (0, 1000))      # Bas

draw = ImageDraw.Draw(new_img)

# Polices
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_legend = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except Exception as e:
    print("Fonts not found:", e)
    font_title = ImageFont.load_default()
    font_legend = font_title

def replace_text(box, text, font, fill_color="white", text_color="black"):
    draw.rectangle(box, fill=fill_color)
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x_center - w/2, y_center - h/2 - 4), text, fill=text_color, font=font)

# Titres traduits
replace_text((0, 0, 900, 60), "Aggregate Score (Agg) Variability", font_title)
replace_text((900, 0, 1800, 60), "k1 to k4 Components Variability", font_title)
replace_text((450, 500, 1350, 560), "Recall Variability", font_title)
replace_text((0, 1000, 1800, 1060), "Spline Function Variability (Average Y Prediction μ per score)", font_title)

# Labels X et Y
replace_text((0, 1445, 1800, 1500), "Transition Score x ∈ {0, 1, 2, 3, 4}", font_legend)

out_path = img_path.replace('.png', '_2.png')
new_img.save(out_path)
print(f"Saved correctly cropped and translated image to {out_path}")
