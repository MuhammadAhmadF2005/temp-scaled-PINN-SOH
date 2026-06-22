import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model

def main():
    results_path = "results.txt"
    if os.path.exists(results_path):
        try:
            os.remove(results_path)
            print("Cleared existing results.txt")
        except Exception as e:
            print(f"Error removing results.txt: {e}")
            
    experiments = [
        {"name": "Holdout 35C - Unscaled", "holdout_temp": 35.0, "use_scaling": False, "plot_dir": "plots_holdout_35_unscaled"},
        {"name": "Holdout 35C - Scaled", "holdout_temp": 35.0, "use_scaling": True, "plot_dir": "plots_holdout_35_scaled"},
        {"name": "Holdout 45C - Unscaled", "holdout_temp": 45.0, "use_scaling": False, "plot_dir": "plots_holdout_45_unscaled"},
        {"name": "Holdout 45C - Scaled", "holdout_temp": 45.0, "use_scaling": True, "plot_dir": "plots_holdout_45_scaled"}
    ]
    
    for i, exp in enumerate(experiments):
        print(f"\n==================================================")
        print(f"Running Experiment {i+1}/{len(experiments)}: {exp['name']}")
        print(f"==================================================")
        
        with open(results_path, "a") as f:
            f.write(f"\n=========================================\n")
            f.write(f"Experiment: {exp['name']}\n")
            f.write(f"Holdout Temp: {exp['holdout_temp']} C, Scaling: {exp['use_scaling']}\n")
            f.write(f"=========================================\n")
            
        train_model(
            data_dir='Dataset_1_NCA_battery',
            epochs=500,
            batch_size=128,
            patience=100,
            use_scaling=exp['use_scaling'],
            holdout_temp=exp['holdout_temp'],
            plot_dir=exp['plot_dir']
        )
        
    print("\nAll experiments completed successfully!")

if __name__ == '__main__':
    main()
