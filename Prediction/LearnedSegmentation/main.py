import sys
from pathlib import Path
import argparse
import datetime
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import json

# Add parent directory to path to import GNN modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'GNN'))

from config_parser import ConfigParser
from load_data import DataLoader
from train import Trainer
from test import Tester
from GNN.arborescence import root_target

def main():
    parser = argparse.ArgumentParser(description="Learned Segmentation Pipeline")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    # 1. Parse Config
    print("Parsing configuration...")
    config = ConfigParser(args.config)
    
    train_flag = config.get_train_flag()
    test_flag = config.get_test_flag()
    
    pipeline_params = config.get_pipeline_params()
    target_variable = config.get_target_variable()
    
    scale = pipeline_params.get('scale', 0)
    tol = pipeline_params.get('tol', 0.3)
    attempt = pipeline_params.get('attempt', 10)
    reduce_param = pipeline_params.get('reduce', 100)
    
    # Construct Experiment Directory
    exp_dir_name = f"target_{target_variable}_scale_{scale}_tol_{tol}_attempt_{attempt}_reduce_{reduce_param}"
    dir_experiment = Path.cwd() / 'Experiments' / exp_dir_name
    dir_experiment.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment Directory: {dir_experiment}")
    
    current_run_dir = None
    model_save_path = None
    model_params = config.get_model_params()
    model_type = model_params.get('type', 'XGBRegressor')
    model_filename = 'learned_segmentation_model.pth' if model_type == 'UNet' else 'learned_segmentation_model.pkl'

    loader = DataLoader(config)
    
    # 2. Train Logic
    # 2. Train Logic
    current_run_dir = dir_experiment / model_type
    
    if train_flag:
        # Create new run directory (or overwrite)
        current_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting run in: {current_run_dir}")
        
        # Load Data
        print("Loading training data...")
        train_depts = config.get_train_departements()
        X_train, y_train = loader.load_all_data(train_depts, fit_scaler=True, fit_encoder=True)
        
        if X_train is None:
            print("ERROR: Failed to load training data.")
            return

        # Save Preprocessed Data
        loader.save_preprocessed_data(current_run_dir, X_train, y_train)
        
        # Update model_params with data shape
        n_channels = X_train.shape[1]
        print(f"Detected {n_channels} input channels.")
        if 'params' not in model_params:
            model_params['params'] = {}
        model_params['params']['n_channels'] = n_channels
        
        # Train
        print("Initializing trainer...")
        trainer = Trainer(model_params)
        trainer.train(X_train, y_train)
        
        # Save Model
        model_save_path = current_run_dir / model_filename
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
        model_save_path = current_run_dir / model_filename
        
        if not model_save_path.exists():
            print(f"ERROR: Model file not found in {current_run_dir}")
            return
             
        # Load Preprocessed Data (to restore scaler/encoder state)
        # We also need to know n_channels to instantiate the model for testing (if needed)
        # Although Tester loads the model, if it's a state_dict, Tester needs to know architecture.
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

    # 3. Test Logic
    if test_flag:
        print("Loading test data...")
        test_depts = config.get_test_departements()
        X_test, y_test = loader.load_all_data(test_depts, fit_scaler=False, fit_encoder=False)
        
        if X_test is not None:
            print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
            tester = Tester(model_save_path, model_params)
            predictions = tester.test(X_test, y_test, compute_metrics=True)
            
            # Visualize
            vis_path = current_run_dir / 'prediction_vis.png'
            tester.visualize_prediction(predictions, y_test, vis_path)
        else:
            print("WARNING: No test data found or loaded.")
            
        # New Test Departments
        new_test_depts = config.get_new_test_departements()
        if new_test_depts:
            print(f"Loading new test data (no scoring) for: {new_test_depts}")
            X_new, y_new = loader.load_all_data(new_test_depts, fit_scaler=False, fit_encoder=False, require_target=False)
            
            if X_new is not None:
                if 'tester' not in locals():
                    tester = Tester(model_save_path, model_params)
                
                predictions_new = tester.test(X_new, y_new, compute_metrics=False)
                
                vis_path_new = current_run_dir / 'prediction_vis_new.png'
                tester.visualize_prediction(predictions_new, y_new, vis_path_new)
            else:
                print("WARNING: No new test data found or loaded.")

if __name__ == "__main__":
    main()
