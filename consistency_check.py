"""
Consistency check: run the model multiple times and compare results.
Runs 3 additional instances for both scaling and no-scaling configurations,
on both holdout and random split, and writes all results to consistency_results.txt.

Updated to use the new variance-reduction pipeline:
- SWA (Stochastic Weight Averaging)
- Cosine Annealing with Warm Restarts
- Snapshot Ensemble at inference
- MC Dropout at inference
- Weight Decay + Gradient Clipping
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim.swa_utils import AveragedModel
from data_loader import get_dataloaders
from model import BatteryPINN, compute_pinn_losses


def train_and_evaluate(data_dir, use_scaling, holdout_temp, epochs=500, batch_size=128, patience=150):
    """Train model with variance-reduction techniques and return test metrics dict."""
    train_loader, val_loader, test_loader, soh_min, soh_range = get_dataloaders(
        data_dir, holdout_temp=holdout_temp, batch_size=batch_size, max_train_cycles=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatteryPINN(n_features=16, use_scaling=use_scaling).to(device)
    
    # Same optimizer setup as train.py: weight decay on network params, not Ea
    ea_params = [p for n, p in model.named_parameters() if 'Ea' in n]
    other_params = [p for n, p in model.named_parameters() if 'Ea' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': ea_params, 'lr': 1e-2, 'weight_decay': 0.0}
    ])
    
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # SWA setup
    swa_model = AveragedModel(model)
    swa_start = 200
    swa_active = False
    
    # Snapshot collection at cosine restart points
    cosine_restart_epochs = set()
    T_cur = 50
    epoch_acc = 0
    while epoch_acc < epochs:
        epoch_acc += T_cur
        if epoch_acc <= epochs:
            cosine_restart_epochs.add(epoch_acc)
        T_cur *= 2
    
    snapshots = []
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        
        scheduler.step(epoch + 1)
        
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
        
        # SWA averaging
        if epoch + 1 >= swa_start:
            if not swa_active:
                swa_active = True
            swa_model.update_parameters(model)
        
        # Snapshot at cosine restart points
        if (epoch + 1) in cosine_restart_epochs:
            snapshot_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            snapshots.append(snapshot_state)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Finalize SWA
    if swa_active:
        swa_model.to(device)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
        snapshots.append(swa_state)
    
    # Add best model
    if best_model_state is not None:
        snapshots.append(best_model_state)
    
    if not snapshots and best_model_state is not None:
        snapshots = [best_model_state]
    
    # Evaluate with Snapshot Ensemble + MC Dropout
    n_mc_samples = 20
    all_u_ensemble = []
    all_soh_list = []
    soh_collected = False
    
    for snap_state in snapshots:
        model.load_state_dict(snap_state)
        model.to(device)
        model.eval()
        
        # Enable dropout for MC sampling
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        
        snap_preds = []
        for mc_i in range(n_mc_samples):
            batch_preds = []
            batch_soh = []
            
            with torch.no_grad():
                for features, soh, temp, cycle in test_loader:
                    features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
                    u, _ = model(features, temp, cycle)
                    batch_preds.append(u.cpu().numpy())
                    if not soh_collected:
                        batch_soh.append(soh.cpu().numpy())
            
            snap_preds.append(np.concatenate(batch_preds))
            if not soh_collected:
                all_soh_list.append(np.concatenate(batch_soh))
                soh_collected = True
        
        snap_mean = np.mean(snap_preds, axis=0)
        all_u_ensemble.append(snap_mean)
    
    all_u = np.mean(all_u_ensemble, axis=0) * soh_range + soh_min
    all_soh = all_soh_list[0] * soh_range + soh_min
    
    test_mse = np.mean((all_soh - all_u)**2)
    test_mae = np.mean(np.abs(all_soh - all_u))
    test_rmse = np.sqrt(test_mse)
    r2 = 1 - np.sum((all_soh - all_u)**2) / np.sum((all_soh - np.mean(all_soh))**2)
    
    return {
        'mse': test_mse,
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': r2,
        'ea': model.Ea.item(),
        'n_snapshots': len(snapshots)
    }


def main():
    n_runs = 3
    data_dir = 'Dataset_1_NCA_battery'
    
    configs = [
        {'name': 'Holdout 35C - No Scaling', 'use_scaling': False, 'holdout_temp': 35.0},
        {'name': 'Holdout 35C - Scaling',    'use_scaling': True,  'holdout_temp': 35.0},
        {'name': 'Random Split - No Scaling',  'use_scaling': False, 'holdout_temp': None},
        {'name': 'Random Split - Scaling',     'use_scaling': True,  'holdout_temp': None},
    ]
    
    # Previous baseline results (before variance reduction)
    baseline = [
        {'name': 'Holdout 35C - No Scaling', 'r2_mean': 0.902, 'r2_std': 0.053},
        {'name': 'Holdout 35C - Scaling',    'r2_mean': 0.782, 'r2_std': 0.134},
        {'name': 'Random Split - No Scaling', 'r2_mean': 0.939, 'r2_std': 0.012},
        {'name': 'Random Split - Scaling',    'r2_mean': 0.945, 'r2_std': 0.004},
    ]
    
    all_results = {cfg['name']: [] for cfg in configs}
    
    output_file = 'consistency_results.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CONSISTENCY CHECK - Variance-Reduced Battery PINN SOH Model\n")
        f.write("Methods: SWA + CosineAnnealing + SnapshotEnsemble + MCDropout\n")
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
            
            results = train_and_evaluate(
                data_dir,
                use_scaling=cfg['use_scaling'],
                holdout_temp=cfg['holdout_temp']
            )
            
            all_results[cfg['name']].append(results)
            
            print(f"  MSE: {results['mse']:.4f} | MAE: {results['mae']:.4f} | "
                  f"RMSE: {results['rmse']:.4f} | R2: {results['r2']:.6f} | "
                  f"Ea: {results['ea']:.2f} | Snapshots: {results['n_snapshots']}")
        
        # Write summary for this config
        runs = all_results[cfg['name']]
        base = baseline[cfg_idx]
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"--- {cfg['name']} ---\n")
            f.write(f"Baseline: R2={base['r2_mean']:.3f} +/- {base['r2_std']:.3f}\n")
            for i, r in enumerate(runs):
                f.write(f"  Run {i+1}:   MSE={r['mse']:.4f}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R2={r['r2']:.6f}  Ea={r['ea']:.2f}  Snaps={r['n_snapshots']}\n")
            
            r2s = [r['r2'] for r in runs]
            rmses = [r['rmse'] for r in runs]
            
            f.write(f"  Mean:    R2={np.mean(r2s):.6f}  RMSE={np.mean(rmses):.4f}\n")
            f.write(f"  Std:     R2={np.std(r2s):.6f}  RMSE={np.std(rmses):.4f}\n")
            
            # Compare with baseline
            improvement = base['r2_std'] - np.std(r2s)
            pct = (improvement / base['r2_std']) * 100 if base['r2_std'] > 0 else 0
            f.write(f"  Variance reduction: {pct:.1f}% (std: {base['r2_std']:.4f} -> {np.std(r2s):.4f})\n")
            f.write(f"\n")
    
    # Final summary
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        for cfg_idx, cfg in enumerate(configs):
            runs = all_results[cfg['name']]
            base = baseline[cfg_idx]
            r2s = [r['r2'] for r in runs]
            
            r2_mean = np.mean(r2s)
            r2_std = np.std(r2s)
            
            improvement = base['r2_std'] - r2_std
            pct = (improvement / base['r2_std']) * 100 if base['r2_std'] > 0 else 0
            
            status = "CONSISTENT" if r2_std < 0.05 else "HIGH VARIANCE"
            
            f.write(f"{cfg['name']}:\n")
            f.write(f"  R2 = {r2_mean:.4f} +/- {r2_std:.4f}\n")
            f.write(f"  Baseline: {base['r2_mean']:.3f} +/- {base['r2_std']:.3f}\n")
            f.write(f"  Variance reduction: {pct:.1f}%\n")
            f.write(f"  Status: {status}\n\n")
    
    print(f"\n\nResults written to {output_file}")
    print("Done!")


if __name__ == '__main__':
    main()
