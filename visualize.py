import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import prepare_dataset, extract_temperature, compute_features
from model import BatteryPINN

def visualize(data_dir, model, plot_dir='plots_scaled', holdout_temp=None):
    print("Preparing dataset (loading files)...")
    train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range = prepare_dataset(data_dir, holdout_temp=holdout_temp, max_train_cycles=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.eval()
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. SOH Trajectories
    print("Plotting SOH trajectories for test cells...")
    temp_batches = {}  # For t_arr plot
    
    for file in test_files:
        temp = extract_temperature(os.path.basename(file))
        df = pd.read_csv(file)
        
        cycles = df['cycle number'].unique()
        cycles = sorted(cycles)
        
        cell_cycles = []
        cell_soh_true = []
        cell_soh_pred = []
        cell_t_arr = []
        
        baseline_soh = None
        valid_cycle_idx = 0
        for cycle_num in cycles:
            df_c = df[df['cycle number'] == cycle_num]
            soh = df_c['Q discharge/mA.h'].max()
            
            if soh <= 10.0:
                continue
            
            # Set baseline from first valid cycle
            if baseline_soh is None:
                baseline_soh = soh
            
            # Skip anomalous cycles (rest/diagnostic) whose capacity
            # drops below 50% of the first cycle's capacity
            if soh < baseline_soh * 0.5:
                continue
            
            features = compute_features(df_c)
            # Normalize
            feat_norm = 2 * ((features - f_min) / f_range) - 1.0
            
            # To tensor
            x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
            temp_t = torch.tensor([[temp]], dtype=torch.float32).to(device)
            cycle_t = torch.tensor([[valid_cycle_idx + 1]], dtype=torch.float32).to(device)
            
            # Enable dropout for Monte Carlo sampling
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    
            n_mc_samples = 30
            u_preds = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    u_pred, t_arr = model(x_t, temp_t, cycle_t)
                    u_preds.append(u_pred.item())
            
            # Revert model to eval mode
            model.eval()
            
            u_pred_val = np.mean(u_preds) * soh_range + soh_min
            
            cell_cycles.append(valid_cycle_idx + 1)
            cell_soh_true.append(soh)
            cell_soh_pred.append(u_pred_val)
            cell_t_arr.append(t_arr.item())
            valid_cycle_idx += 1
        
        if temp not in temp_batches:
            temp_batches[temp] = []
        temp_batches[temp].append((cell_cycles, cell_t_arr))
        
        # Plot SOH
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
        
    # 2. Thermal Age vs Cycle Number
    print("Plotting Thermal Age (t_arr) vs Cycle Number...")
    plt.figure(figsize=(8, 5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    c_idx = 0
    for temp, lst in temp_batches.items():
        # Just plot the first cell of this temperature as representative
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

