"""
Exemple d'utilisation des fonctionnalités SHAP améliorées

Ce script démontre comment utiliser les nouvelles fonctions SHAP pour :
1. Calculer et sauvegarder les SHAP values pour tout le dataset
2. Analyser des échantillons spécifiques sans recalculer
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Ajouter le dossier parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GNN.tools import read_object
from GNN.pytorch_model import Training, WrapperModel

def load_model_and_data():
    """
    Charge le modèle spécifié et tente de récupérer les données de test
    """
    model_path = Path("/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_default/check_z-score/full_all_4_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node/GRU_full_full_10_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss/GRU_full_full_10_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss.pkl")
    
    print(f"Chargement du modèle depuis : {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Le fichier modèle n'existe pas : {model_path}")
        
    model = read_object(model_path.name, model_path.parent)
    print(f"Modèle chargé : {type(model)}")
    
    # Essayer de récupérer les données de test
    # Option 1: df_test stocké dans le modèle
    if hasattr(model, 'df_test'):
        print("Utilisation de model.df_test")
        return model, model.df_test
        
    # Option 2: test_loader stocké dans le modèle
    if hasattr(model, 'test_loader'):
        print("Extraction des données depuis model.test_loader")
        # Reconstruire un DataFrame à partir du loader
        # Ceci est une approximation, idéalement on voudrait le DataFrame original
        # Mais pour SHAP, on a besoin des features
        
        # Note: Cette partie dépend de la structure exacte de votre DataLoader
        # Voici une tentative générique
        data_list = []
        for batch in model.test_loader:
            # Supposons que le batch contient [inputs, labels, ...]
            # Adaptez selon votre collate_fn
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                # Si inputs est un tenseur, on le convertit
                if torch.is_tensor(inputs):
                    data_list.append(inputs.cpu().numpy())
            elif isinstance(batch, dict):
                # Si c'est un dictionnaire
                pass
                
        if data_list:
            X = np.concatenate(data_list, axis=0)
            # Créer un DataFrame factice avec les noms de features
            if hasattr(model, 'features_name'):
                # Attention: X peut avoir plus de dimensions (B, F, T)
                # Pour le DataFrame, on a besoin de 2D, mais SHAP gère le 3D via WrapperModel
                # Ici on retourne juste X pour l'instant, il faudra adapter shapley_additive_explanation
                # pour accepter un numpy array si df n'est pas fourni
                pass
    
    print("⚠️ Impossible de trouver df_test automatiquement.")
    print("Veuillez fournir le chemin vers le fichier de données de test.")
    # Pour l'instant, on retourne None pour les données
    return model, None

def example_full_shap_analysis(model, test_df, output_dir):
    """
    Exemple 1 : Calculer les SHAP values pour tout le dataset
    """
    print("=" * 80)
    print("EXEMPLE 1 : Calcul complet des SHAP values")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if test_df is None:
        print("❌ Pas de données de test fournies. Impossible de calculer SHAP.")
        return

    # Réduction du dataset à 100 échantillons aléatoires si nécessaire
    if len(test_df) > 100:
        print(f"⚠️ Réduction du dataset : 100 échantillons aléatoires sur {len(test_df)}")
        test_df_pos = test_df.sample(n=1000, random_state=42)
        test_df['weight'] = 0
        test_df.loc[test_df_pos.index, 'weight'] = 1

    # Calculer et sauvegarder les SHAP values
    print("\n📊 Calcul des SHAP values pour tout le dataset...")
    print(f"   Nombre d'échantillons : {len(test_df)}")
    print(f"   Cela peut prendre 10-30 minutes...\n")
    
    model.shapley_additive_explanation(
        df=test_df,
        outname='wildfire_prediction',
        dir_output=output_dir,
        mode='bar',  # ou 'beeswarm'
        figsize=(50, 25)
    )
    
    print("\n✅ SHAP values calculées et sauvegardées !")
    print(f"   Fichier : {output_dir / 'wildfire_prediction_shap_values.pkl'}")
    if (output_dir / 'wildfire_prediction_shap_values.pkl').exists():
        print(f"   Taille : {(output_dir / 'wildfire_prediction_shap_values.pkl').stat().st_size / 1024 / 1024:.2f} MB")


def example_single_sample_analysis(model, output_dir):
    """
    Exemple 2 : Analyser un échantillon spécifique
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 2 : Analyse d'un échantillon spécifique")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    # Analyser l'échantillon 0
    print("\n🔍 Analyse de l'échantillon 0...")
    
    try:
        results = model.shapley_additive_explanation_sample(
            sample_idx=0,
            outname='wildfire_prediction',
            dir_output=output_dir,
            sample_name='high_risk_example',
            figsize=(15, 10),
            generate_force_plot=True
        )
        
        print("\n✅ Analyse terminée !")
        print(f"   Échantillon : {results['sample_name']}")
        print(f"   Nombre de classes : {results['shap_values'].shape[1]}")
        print(f"   Nombre de features : {len(results['feature_names'])}")
        print(f"   Visualisations générées : {len(results['plots_generated'])}")
        
        # Afficher les top features
        print("\n📈 Top 5 features les plus importantes (classe 0) :")
        class_0_shap = np.abs(results['shap_values'][:, 0])
        top_indices = np.argsort(class_0_shap)[::-1][:5]
        
        for rank, idx in enumerate(top_indices, 1):
            feature_name = results['feature_names'][idx]
            shap_value = results['shap_values'][idx, 0]
            feature_value = results['features'][idx]
            print(f"   {rank}. {feature_name:30s} | SHAP: {shap_value:+.4f} | Valeur: {feature_value:.4f}")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de l'échantillon : {e}")


def example_multiple_samples_analysis(model, output_dir, sample_indices):
    """
    Exemple 3 : Analyser plusieurs échantillons
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 3 : Analyse de plusieurs échantillons")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    print(f"\n🔍 Analyse de {len(sample_indices)} échantillons...")
    
    all_results = []
    
    for i, idx in enumerate(sample_indices, 1):
        try:
            print(f"\n   [{i}/{len(sample_indices)}] Échantillon {idx}...", end=" ")
            
            results = model.shapley_additive_explanation_sample(
                sample_idx=idx,
                outname='wildfire_prediction',
                dir_output=output_dir,
                sample_name=f'sample_{idx:04d}',
                generate_force_plot=False  # Désactiver pour gagner du temps
            )
            
            all_results.append(results)
            print(f"✓ ({len(results['plots_generated'])} plots)")
            
        except Exception as e:
            print(f"✗ Erreur : {e}")
    
    print(f"\n✅ {len(all_results)}/{len(sample_indices)} échantillons traités avec succès !")
    
    return all_results


def example_compare_samples(results_list):
    """
    Exemple 4 : Comparer les SHAP values de plusieurs échantillons
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 4 : Comparaison des échantillons")
    print("=" * 80)
    
    if len(results_list) < 2:
        print("⚠️  Besoin d'au moins 2 échantillons pour comparer")
        return
    
    print(f"\n📊 Comparaison de {len(results_list)} échantillons...")
    
    # Créer un DataFrame pour comparer
    comparison_data = []
    
    for results in results_list:
        sample_name = results['sample_name']
        shap_vals = results['shap_values'][:, 0]  # Classe 0
        feature_names = results['feature_names']
        
        for fname, sval in zip(feature_names, shap_vals):
            comparison_data.append({
                'sample': sample_name,
                'feature': fname,
                'shap_value': sval
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Trouver les features les plus variables entre échantillons
    print("\n🔄 Features avec la plus grande variabilité entre échantillons :")
    
    variance_by_feature = df_comparison.groupby('feature')['shap_value'].var().sort_values(ascending=False)
    
    for rank, (feature, variance) in enumerate(variance_by_feature.head(5).items(), 1):
        print(f"   {rank}. {feature:30s} | Variance: {variance:.6f}")
    
    # Afficher les valeurs moyennes
    print("\n📊 SHAP moyen par feature (top 5) :")
    mean_by_feature = df_comparison.groupby('feature')['shap_value'].mean().abs().sort_values(ascending=False)
    
    for rank, (feature, mean_val) in enumerate(mean_by_feature.head(5).items(), 1):
        print(f"   {rank}. {feature:30s} | SHAP moyen: {mean_val:.4f}")


def example_load_and_reuse(output_dir):
    """
    Exemple 5 : Charger et réutiliser les SHAP values sauvegardées
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 5 : Réutilisation des SHAP values sauvegardées")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    shap_file = output_dir / 'wildfire_prediction_shap_values.pkl'
    
    if not shap_file.exists():
        print(f"⚠️  Fichier SHAP introuvable : {shap_file}")
        return
    
    print(f"\n📂 Chargement depuis : {shap_file}")
    
    shap_data = read_object(shap_file.name, shap_file.parent)
    
    print("\n✅ Données chargées !")
    print(f"   Nombre d'échantillons (B) : {shap_data['B']}")
    print(f"   Nombre de features (F) : {shap_data['F']}")
    print(f"   Nombre de pas de temps (T) : {shap_data['T']}")
    print(f"   Nombre de classes : {shap_data['n_classes']}")
    print(f"   Shape des SHAP values : {shap_data['shap_values'].shape}")
    
    # Statistiques globales
    print("\n📊 Statistiques globales :")
    shap_values = shap_data['shap_values']
    
    for class_idx in range(shap_data['n_classes']):
        class_shap = shap_values[:, :, class_idx]
        print(f"\n   Classe {class_idx} :")
        print(f"      SHAP min  : {class_shap.min():.4f}")
        print(f"      SHAP max  : {class_shap.max():.4f}")
        print(f"      SHAP mean : {class_shap.mean():.4f}")
        print(f"      SHAP std  : {class_shap.std():.4f}")


def main():
    """
    Fonction principale pour exécuter tous les exemples
    """
    print("\n" + "=" * 80)
    print("EXEMPLES D'UTILISATION DES FONCTIONNALITÉS SHAP")
    print("=" * 80)
    
    # Configuration
    output_dir = Path('./shap_analysis_examples')
    
    # Charger le modèle et les données
    try:
        model, test_df = load_model_and_data()
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return

    if model is None:
        return

    # Si on n'a pas trouvé de données, on ne peut pas faire l'exemple 1
    if test_df is not None:
        # Exemple 1 : Calcul complet (à faire une seule fois)
        # Note: Commenté par défaut car long
        example_full_shap_analysis(model, test_df, output_dir)
    else:
        print("⚠️ Pas de DataFrame de test trouvé. L'exemple 1 est ignoré.")
        print("   Si vous avez déjà calculé les SHAP values, les exemples suivants fonctionneront.")

    # Exemple 2 : Analyser un échantillon
    # Cela fonctionnera SI les SHAP values ont déjà été calculées et sauvegardées
    example_single_sample_analysis(model, output_dir)
    
    # Exemple 3 : Analyser plusieurs échantillons
    sample_indices = [0, 10, 20, 30, 42]
    results = example_multiple_samples_analysis(model, output_dir, sample_indices)
    
    # Exemple 4 : Comparer les échantillons
    if results:
        example_compare_samples(results)
    
    # Exemple 5 : Charger et réutiliser
    example_load_and_reuse(output_dir)
    
    print("\n" + "=" * 80)
    print("FIN DES EXEMPLES")
    print("=" * 80)


if __name__ == "__main__":
    main()
