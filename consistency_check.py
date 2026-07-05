"""
Consistency check: run the model multiple times and compare results.
Runs 3 additional instances for both scaling and no-scaling configurations,
on both holdout and random split, and writes all results to consistency_results.txt.
"""
import torch
import numpy as np
import sys
import os
from data_loader import get_dataloaders
from model import BatteryPINN, compute_pinn_losses

def train_and_evaluate(data_dir, use_scaling, holdout_temp, epochs=500, batch_size=128, patience=100):
    """Train model and return test metrics dict."""
    train_loader, val_loader, test_loader, soh_min, soh_range = get_dataloaders(
        data_dir, holdout_temp=holdout_temp, batch_size=batch_size, max_train_cycles=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatteryPINN(n_features=16, use_scaling=use_scaling).to(device)
    
    # Same optimizer setup as train.py
    ea_params = [p for n, p in model.named_parameters() if 'Ea' in n]
    other_params = [p for n, p in model.named_parameters() if 'Ea' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3},
        {'params': ea_params, 'lr': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, soh, temp, cycle in train_loader:
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            features.requires_grad_(True)
            optimizer.zero_grad()
            
            loss, mse_loss, pde_loss, mono_loss = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.01, beta=0.1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        
        model.eval()
        val_loss = 0.0
        with torch.enable_grad():
            for features, soh, temp, cycle in val_loader:
                features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
                features.requires_grad_(True)
                loss, _, _, _ = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.01, beta=0.1)
                val_loss += loss.item() * features.size(0)
        
        n_val = len(val_loader.dataset)
        val_loss /= n_val
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Ea={model.Ea.item():.2f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    all_u = []
    all_soh = []
    
    with torch.no_grad():
        for features, soh, temp, cycle in test_loader:
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            u, _ = model(features, temp, cycle)
            all_u.append(u.cpu().numpy())
            all_soh.append(soh.cpu().numpy())
    
    all_u = np.concatenate(all_u) * soh_range + soh_min
    all_soh = np.concatenate(all_soh) * soh_range + soh_min
    
    test_mse = np.mean((all_soh - all_u)**2)
    test_mae = np.mean(np.abs(all_soh - all_u))
    test_rmse = np.sqrt(test_mse)
    r2 = 1 - np.sum((all_soh - all_u)**2) / np.sum((all_soh - np.mean(all_soh))**2)
    
    return {
        'mse': test_mse,
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': r2,
        'ea': model.Ea.item()
    }


def main():
    n_runs = 3
    data_dir = 'Dataset_1_NCA_battery'
    
    configs = [
        {'name': 'Holdout 35°C - No Scaling', 'use_scaling': False, 'holdout_temp': 35.0},
        {'name': 'Holdout 35°C - Scaling',    'use_scaling': True,  'holdout_temp': 35.0},
        {'name': 'Random Split - No Scaling',  'use_scaling': False, 'holdout_temp': None},
        {'name': 'Random Split - Scaling',     'use_scaling': True,  'holdout_temp': None},
    ]
    
    # Reference results from results.txt
    reference = [
        {'name': 'Holdout 35°C - No Scaling', 'mse': 5986.26, 'mae': 73.47, 'rmse': 77.37, 'r2': -1.76},
        {'name': 'Holdout 35°C - Scaling',    'mse': 674.26,  'mae': 21.24, 'rmse': 25.97, 'r2': 0.689},
        {'name': 'Random Split - No Scaling',  'mse': 1500.67, 'mae': 27.42, 'rmse': 38.74, 'r2': 0.953},
        {'name': 'Random Split - Scaling',     'mse': 633.19,  'mae': 18.40, 'rmse': 25.16, 'r2': 0.980},
    ]
    
    all_results = {cfg['name']: [] for cfg in configs}
    
    output_file = 'consistency_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CONSISTENCY CHECK - Multiple Runs of Battery PINN SOH Model\n")
        f.write("=" * 70 + "\n\n")
    
    for cfg_idx, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config: {cfg['name']}")
        print(f"{'='*60}")
        
        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} ---")
            # Different random seeds for each run to test consistency
            torch.manual_seed(run * 100 + 42)
            np.random.seed(run * 100 + 42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run * 100 + 42)
            
            # But we need data loader to use seed 42 for consistent splits
            # The data_loader.py already uses np.random.seed(42) internally
            # We just need to make sure model init varies
            
            results = train_and_evaluate(
                data_dir,
                use_scaling=cfg['use_scaling'],
                holdout_temp=cfg['holdout_temp']
            )
            
            all_results[cfg['name']].append(results)
            
            print(f"  MSE: {results['mse']:.4f} | MAE: {results['mae']:.4f} | "
                  f"RMSE: {results['rmse']:.4f} | R²: {results['r2']:.6f} | Ea: {results['ea']:.2f}")
        
        # Write summary for this config
        runs = all_results[cfg['name']]
        ref = reference[cfg_idx]
        
        with open(output_file, 'a') as f:
            f.write(f"--- {cfg['name']} ---\n")
            f.write(f"Reference:  MSE={ref['mse']:.2f}  MAE={ref['mae']:.2f}  RMSE={ref['rmse']:.2f}  R²={ref['r2']:.3f}\n")
            for i, r in enumerate(runs):
                f.write(f"  Run {i+1}:   MSE={r['mse']:.4f}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R²={r['r2']:.6f}  Ea={r['ea']:.2f}\n")
            
            mses = [r['mse'] for r in runs]
            maes = [r['mae'] for r in runs]
            rmses = [r['rmse'] for r in runs]
            r2s = [r['r2'] for r in runs]
            eas = [r['ea'] for r in runs]
            
            f.write(f"  Mean:    MSE={np.mean(mses):.4f}  MAE={np.mean(maes):.4f}  RMSE={np.mean(rmses):.4f}  R²={np.mean(r2s):.6f}  Ea={np.mean(eas):.2f}\n")
            f.write(f"  Std:     MSE={np.std(mses):.4f}  MAE={np.std(maes):.4f}  RMSE={np.std(rmses):.4f}  R²={np.std(r2s):.6f}  Ea={np.std(eas):.2f}\n")
            f.write(f"\n")
    
    # Final summary
    with open(output_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL CONSISTENCY SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        for cfg_idx, cfg in enumerate(configs):
            runs = all_results[cfg['name']]
            ref = reference[cfg_idx]
            r2s = [r['r2'] for r in runs]
            rmses = [r['rmse'] for r in runs]
            
            r2_mean = np.mean(r2s)
            r2_std = np.std(r2s)
            rmse_mean = np.mean(rmses)
            rmse_std = np.std(rmses)
            
            # Check if results are within reasonable range of reference
            r2_diff = abs(r2_mean - ref['r2'])
            
            status = "✓ CONSISTENT" if r2_std < 0.05 else "⚠ HIGH VARIANCE"
            
            f.write(f"{cfg['name']}:\n")
            f.write(f"  R² = {r2_mean:.4f} ± {r2_std:.4f}  (ref: {ref['r2']:.3f}, diff: {r2_diff:.4f})\n")
            f.write(f"  RMSE = {rmse_mean:.2f} ± {rmse_std:.2f}  (ref: {ref['rmse']:.2f})\n")
            f.write(f"  Status: {status}\n\n")
    
    print(f"\n\nResults written to {output_file}")
    print("Done!")


if __name__ == '__main__':
    main()
