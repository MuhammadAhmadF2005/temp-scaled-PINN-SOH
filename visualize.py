import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import prepare_dataset, extract_temperature, compute_features
from model import BatteryPINN

def visualize(data_dir, models, plot_dir='plots_scaled', holdout_temp=None):
    """
    Visualize SOH predictions using snapshot ensemble + MC Dropout.
    
    Args:
        data_dir: Path to dataset directory
        models: List of BatteryPINN models (snapshot ensemble) or a single model
        plot_dir: Directory to save plots
        holdout_temp: Temperature held out for testing (or None for random split)
    """

    if not isinstance(models, list):
        models = [models]
    
    print(f"Preparing dataset (loading files)...")
    train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range = prepare_dataset(data_dir, holdout_temp=holdout_temp, max_train_cycles=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using {len(models)} snapshot model(s) for ensemble prediction")
    
    for m in models:
        m.eval()
    
    os.makedirs(plot_dir, exist_ok=True)
    
    n_mc_samples = 30
    

    print("Plotting SOH trajectories for test cells...")
    temp_batches = {}
    
    for file in test_files:
        temp = extract_temperature(os.path.basename(file))
        df = pd.read_csv(file)
        
        cycles = df['cycle number'].unique()
        cycles = sorted(cycles)
        
        cell_cycles = []
        cell_soh_true = []
        cell_soh_pred = []
        cell_t_arr = []
        

        valid_raw_cycles = []
        baseline_soh = None
        for cycle_num in cycles:
            df_c = df[df['cycle number'] == cycle_num]
            soh = df_c['Q discharge/mA.h'].max()
            
            if soh <= 10.0:
                continue
            

            if baseline_soh is None:
                baseline_soh = soh
            


            if soh < baseline_soh * 0.5:
                continue
            
            valid_raw_cycles.append({
                'df_c': df_c,
                'soh': soh
            })
            
        if not valid_raw_cycles:
            continue
            

        soh_raw_vals = [c['soh'] for c in valid_raw_cycles]
        s_series = pd.Series(soh_raw_vals)
        soh_smoothed = s_series.rolling(window=5, center=True, min_periods=1).median().values
        

        for idx, c in enumerate(valid_raw_cycles):
            features = compute_features(c['df_c'])

            feat_norm = 2 * ((features - f_min) / f_range) - 1.0
            

            x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
            temp_t = torch.tensor([[temp]], dtype=torch.float32).to(device)
            cycle_t = torch.tensor([[idx + 1]], dtype=torch.float32).to(device)
            

            all_preds = []
            last_t_arr = None
            
            for snap_model in models:

                snap_model.eval()
                for m in snap_model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                
                with torch.no_grad():
                    for _ in range(n_mc_samples):
                        u_pred, t_arr = snap_model(x_t, temp_t, cycle_t)
                        all_preds.append(u_pred.item())
                        last_t_arr = t_arr.item()
                

                snap_model.eval()
            

            u_pred_val = np.mean(all_preds) * soh_range + soh_min
            
            cell_cycles.append(idx + 1)
            cell_soh_true.append(soh_smoothed[idx])
            cell_soh_pred.append(u_pred_val)
            cell_t_arr.append(last_t_arr)
        
        if temp not in temp_batches:
            temp_batches[temp] = []
        temp_batches[temp].append((cell_cycles, cell_t_arr))
        

        plt.figure(figsize=(8, 5))
        plt.plot(cell_cycles, cell_soh_true, label='True SOH', color='black', linewidth=2)
        plt.plot(cell_cycles, cell_soh_pred, label='Predicted SOH', color='red', linestyle='dashed')
        plt.xlabel('Cycle')
        plt.ylabel('SOH (Max Discharge Capacity / mA.h)')
        plt.title(f'SOH Trajectory - {os.path.basename(file)} (Temp: {temp}°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        safe_filename = os.path.basename(file).replace('#', 'num')
        plt.savefig(f"{plot_dir}/soh_{safe_filename}.png")
        plt.close()
        

    print("Plotting Thermal Age (t_arr) vs Cycle Number...")
    plt.figure(figsize=(8, 5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    c_idx = 0
    for temp, lst in temp_batches.items():

        if len(lst) > 0:
            c, t_arr = lst[0]
            plt.plot(c, t_arr, label=f'{temp}°C', color=colors[c_idx % len(colors)], linewidth=2)
            c_idx += 1
            
    plt.xlabel('Cycle Number (t)')
    plt.ylabel('Equivalent Aging Time (t_arr)')
    plt.title('Thermal Age vs Cycle Number across Temperatures')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/thermal_age_vs_cycle.png')
    plt.close()

#python file for the purspose of visualizing plots...
