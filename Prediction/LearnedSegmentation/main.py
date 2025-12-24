import sys
from pathlib import Path
import argparse
import datetime
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import json
import logging
import numpy as np

# Add parent directory to path to import GNN modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'GNN'))

from config_parser import ConfigParser
from load_data import DataLoader
from train import Trainer
from test import Tester
from GNN.arborescence import root_target
from GNN.construct import parse_string

def main():
    parser = argparse.ArgumentParser(description="Learned Segmentation Pipeline")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    # 1. Parse Config
    print("Parsing configuration...")
    config = ConfigParser(args.config)
    
    train_flag = config.get_train_flag()
    test_flag = config.get_test_flag()
    # 1. Experiment Directory
    model_params = config.get_model_params() # Keep this line to get model_params for later use
    model_type = model_params.get('type', 'XGBRegressor') # Get model_type from model_params
    target_variable = config.get_target_variable()
    target_type = config.get_target_type()
    
    scale = config.get_scale()
    assert scale is not None, "Scale must be specified in config"
    
    graph_construct = config.get_graph_construct()
    assert graph_construct is not None, "Graph construct must be specified in config"
        
    dico_config = parse_string(graph_construct)

    tol = dico_config.get('tol') # Revert to original default
    attempt = dico_config.get('attempt') # Revert to original default
    reduce_param = dico_config.get('reduce') # Revert to original default variable name and default value
    
    assert attempt is not None, "Attempt must be specified in config"
    assert reduce_param is not None, "Reduce must be specified in config"
    assert tol is not None, "Tol must be specified in config"
    
    n_clusters_node = config.get_n_clusters_node()

    # Construct Experiment Directory
    exp_dir_name = f"target_{target_variable}_{target_type}_scale_{scale}_tol_{tol}_attempt_{attempt}_reduce_{reduce_param}_ncluster_{n_clusters_node}"
    dir_experiment = Path.cwd() / 'Experiments' / exp_dir_name
    dir_experiment.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment Directory: {dir_experiment}")
    
    current_run_dir = None
    model_save_path = None
    model_params = config.get_model_params()
    model_type = model_params.get('type', None)
    assert model_type is not None, "Model type must be specified in config"
    model_filename = 'learned_segmentation_model.pth' if model_type == 'UNet' else 'learned_segmentation_model.pkl'

    loader = DataLoader(config)
    
    # 2. Train Logic
    current_run_dir = dir_experiment / model_type
    
    if train_flag:
        # Create new run directory (or overwrite)
        current_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting run in: {current_run_dir}")
        
        # Load Data
        print("Loading training data...")
        train_depts = config.get_train_departements()
        
        # Fit Clusterer if needed (Global step before loading individual dept data)
        target_type = config.get_target_type()
        loader.launch_segmentation(train_depts)
             
        X_train, y_train, weights_train = loader.load_all_data(train_depts, load=True, fit_scaler=True)

        loader.save_preprocessed_data(dir_experiment, X_train, y_train, weights_train)
        
        if X_train is None:
            print("ERROR: Failed to load training data.")
            return

        # Update model_params with data shape
        n_channels = X_train.shape[1]
        print(f"Detected {n_channels} input channels.")
        if 'params' not in model_params:
            raise ValueError("Model parameters must contain 'params' key.")

        model_params['params']['n_channels'] = n_channels

        if model_params['params']['task_type'] == 'classification' or model_params['params']['task_type'] == 'binary':
            y_train = y_train.astype(int)
        
        # Train
        print("Initializing trainer...")
        trainer = Trainer(model_params)
        trainer.train(X_train, y_train, weights=weights_train)
        
        # Plot Loss
        trainer.plot_loss(current_run_dir / 'loss_plot.png')
        
        # Save Model
        model_save_path = current_run_dir / f'{model_filename}_{model_params["params"]["task_type"]}.pth'
        trainer.save_model(model_save_path)
        
        # Save model params used for training
        with open(current_run_dir / 'model_params.json', 'w') as f:
            json.dump(model_params, f, indent=4)
        
    else:
        # Load existing run
        if not current_run_dir.exists():
             print(f"ERROR: Run directory not found: {current_run_dir}")
             return
             
        print(f"Loading run from: {current_run_dir}")
        model_save_path = current_run_dir / f'{model_filename}_{model_params["params"]["task_type"]}.pth'
        if not model_save_path.exists():
            model_save_path = current_run_dir / model_filename
        
        if not model_save_path.exists():
            print(f"ERROR: Model file not found in {current_run_dir}")
            return
             
        # Let's load model_params from file if it exists
        if (current_run_dir / 'model_params.json').exists():
            with open(current_run_dir / 'model_params.json', 'r') as f:
                model_params = json.load(f)
            print("Loaded model params from file.")
        else:
             # Fallback: try to infer from loaded data
             X_loaded, _ = loader.load_preprocessed_data(current_run_dir)
             if X_loaded is not None:
                 model_params['params']['n_channels'] = X_loaded.shape[1]
             else:
                 print("WARNING: Could not infer n_channels. Using config value (might be wrong).")

        # Ensure loader state is restored
        if not (current_run_dir / 'scaler.pkl').exists():
             # If we didn't load preprocessed data above, do it now
             loader.load_preprocessed_data(current_run_dir)
             
             # Fallback to experiment directory if scaler not found in run directory
             # Check if scaler is fitted (has mean_ attribute)
             if not hasattr(loader.scaler, 'mean_') and (dir_experiment / 'scaler.pkl').exists():
                 print(f"Loading scaler from experiment directory: {dir_experiment}")
                 loader.load_preprocessed_data(dir_experiment)

    # 3. Test Logic
    if test_flag:
        print("Loading test data...")
        test_depts = config.get_test_departements()

        # 3.1 Test on Training Set
        print("\n--- Testing on Training Set ---")
        train_depts = config.get_train_departements()
        train_output_dir = current_run_dir / 'train'
        train_output_dir.mkdir(parents=True, exist_ok=True)
        
        for dept in train_depts:
            print(f"\nTesting training department: {dept}")
            X_train_test, y_train_test, weights_train_test = loader.load_all_data([dept], load=False, fit_scaler=False)
            
            if X_train_test is not None:
                if 'tester' not in locals():
                     tester = Tester(config, model_save_path, model_params)
                
                tester.test(X_train_test, y_train_test, weights=weights_train_test, dept_name=dept, compute_metrics=True, output_dir=train_output_dir)
            else:
                print(f"WARNING: No training data found or loaded for {dept} during testing phase.")

        # 3.2 Test on Test Set
        print("\n--- Testing on Test Set ---")
        test_output_dir = current_run_dir / 'test'
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate over each test department
        for dept in test_depts:
            print(f"\nTesting department: {dept}")
            # Load data for single department
            X_test, y_test, weights_test = loader.load_all_data([dept], load=False, fit_scaler=False)
            if X_test is not None:
                print(f"Test data shape for {dept}: X={X_test.shape}, y={y_test.shape}")

                if 'tester' not in locals():
                     tester = Tester(config, model_save_path, model_params)
                
                predictions = tester.test(X_test, y_test, weights=weights_test, dept_name=dept, compute_metrics=True, output_dir=test_output_dir)
                
            else:
                print(f"WARNING: No test data found or loaded for {dept}.")
        
        if 'tester' in locals():
             tester.save_dataframe(current_run_dir / 'scores.csv')
            
        # New Test Departments
        new_test_depts = config.get_new_test_departements()
        if new_test_depts:
            print(f"\nLoading new test data (no scoring) for: {new_test_depts}")
            new_test_output_dir = current_run_dir / 'new_test'
            new_test_output_dir.mkdir(parents=True, exist_ok=True)
            
            for dept in new_test_depts:
                print(f"\nTesting new department: {dept}")
                X_new, y_new, weights_new = loader.load_all_data([dept], require_target=False, load=False, fit_scaler=False)
                
                if X_new is not None:
                    if 'tester' not in locals():
                        tester = Tester(config, model_save_path, model_params)
                    
                    predictions_new = tester.test(X_new, y_new, weights=weights_new, dept_name=dept, compute_metrics=False, output_dir=new_test_output_dir)
                    
                else:
                    print(f"WARNING: No new test data found or loaded for {dept}.")

if __name__ == "__main__":
    main()
