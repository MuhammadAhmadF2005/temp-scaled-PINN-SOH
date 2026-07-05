import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
from torch.optim.swa_utils import AveragedModel, SWALR
from data_loader import get_dataloaders
from model import BatteryPINN, compute_pinn_losses
from visualize import visualize

def train_model(data_dir, epochs=500, batch_size=128, patience=150, use_scaling=True, holdout_temp=None, plot_dir='plots_scaled'):
    print("Loading data...")
    train_loader, val_loader, test_loader, soh_min, soh_range = get_dataloaders(data_dir, holdout_temp=holdout_temp, batch_size=batch_size, max_train_cycles=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Initializing model (use_scaling={use_scaling})...")
    model = BatteryPINN(n_features=16, use_scaling=use_scaling).to(device)
    
    # Differential learning rates: Ea gets 1e-2 (no weight decay), others get 1e-3 with L2 regularization
    ea_params = [p for n, p in model.named_parameters() if 'Ea' in n]
    other_params = [p for n, p in model.named_parameters() if 'Ea' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': ea_params, 'lr': 1e-2, 'weight_decay': 0.0}
    ])
    
    # Cosine Annealing with Warm Restarts: T_0=50 epochs, doubles each restart
    # Restart schedule: epochs 50, 150, 350, (500+ would be next)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # SWA setup: starts averaging weights after epoch 200
    swa_model = AveragedModel(model)
    swa_start = 200
    swa_active = False
    
    # Snapshot ensemble: save model at cosine restart convergence points
    # With T_0=50, T_mult=2: restarts at epochs 50, 150, 350
    # We save just before each restart (at the converged point)
    cosine_restart_epochs = set()
    T_cur = 50
    epoch_acc = 0
    while epoch_acc < epochs:
        epoch_acc += T_cur
        if epoch_acc <= epochs:
            cosine_restart_epochs.add(epoch_acc)
        T_cur *= 2
    print(f"Cosine restart epochs (snapshot points): {sorted(cosine_restart_epochs)}")
    
    snapshots = []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_pde = 0.0
        train_mono = 0.0
        
        for batch_i, (features, soh, temp, cycle) in enumerate(train_loader):
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            features.requires_grad_(True)
            optimizer.zero_grad()
            
            loss, mse_loss, pde_loss, mono_loss = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.01, beta=0.1)
            
            loss.backward()
            
            # Gradient clipping to prevent gradient spikes from PDE/mono losses
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            train_mse += mse_loss.item() * features.size(0)
            train_pde += pde_loss.item() * features.size(0)
            train_mono += mono_loss.item() * features.size(0)
            
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_mse /= n_train
        train_pde /= n_train
        train_mono /= n_train
        
        # Step cosine annealing scheduler
        scheduler.step(epoch + 1)
        
        model.eval()
        val_loss = 0.0
                
        # We will compute full PINN validation loss using enable_grad()
        with torch.enable_grad():
            for features, soh, temp, cycle in val_loader:
                features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
                features.requires_grad_(True)
                loss, _, _, _ = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.01, beta=0.1)
                val_loss += loss.item() * features.size(0)
        
        n_val = len(val_loader.dataset)
        val_loss /= n_val
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} [MSE:{train_mse:.6f} PDE:{train_pde:.6f} MONO:{train_mono:.6f}] | Val Loss: {val_loss:.6f} | Ea = {model.Ea.item():.2f}")
        
        # SWA: start averaging after swa_start epochs
        if epoch + 1 >= swa_start:
            if not swa_active:
                print(f"  --> SWA activated at epoch {epoch+1}")
                swa_active = True
            swa_model.update_parameters(model)
        
        # Snapshot: save model state at cosine restart convergence points
        if (epoch + 1) in cosine_restart_epochs:
            snapshot_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            snapshots.append(snapshot_state)
            print(f"  --> Snapshot #{len(snapshots)} saved at epoch {epoch+1}")
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Saved new best model to memory")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    print("Training finished.")
    
    # Finalize SWA model: update batch norm statistics
    if swa_active:
        print("Updating SWA batch norm statistics...")
        # SWA model needs BN update with training data
        swa_model.to(device)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        # Add SWA model as a snapshot
        swa_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
        snapshots.append(swa_state)
        print(f"  --> SWA model added as snapshot #{len(snapshots)}")
    
    # Always include the best early-stopped model as a snapshot
    if best_model_state is not None:
        snapshots.append(best_model_state)
        print(f"  --> Best model added as snapshot #{len(snapshots)}")
    
    # If no snapshots were collected (e.g. very early stopping), use best model
    if not snapshots and best_model_state is not None:
        snapshots = [best_model_state]
    
    print(f"\nEvaluating on test set with {len(snapshots)} snapshot(s) + MC Dropout (20 samples each)...")
    
    n_mc_samples = 20
    all_u_ensemble = []
    all_soh_list = []
    soh_collected = False
    
    for snap_idx, snap_state in enumerate(snapshots):
        model.load_state_dict(snap_state)
        model.to(device)
        model.eval()
        
        # MC Dropout: enable dropout during inference
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
        
        # Average MC samples for this snapshot
        snap_mean = np.mean(snap_preds, axis=0)
        all_u_ensemble.append(snap_mean)
    
    # Average across all snapshots
    all_u = np.mean(all_u_ensemble, axis=0) * soh_range + soh_min
    all_soh = all_soh_list[0] * soh_range + soh_min
    
    test_mse = np.mean((all_soh - all_u)**2)
    test_mae = np.mean(np.abs(all_soh - all_u))
    test_rmse = np.sqrt(test_mse)
    
    r2 = 1 - np.sum((all_soh - all_u)**2) / np.sum((all_soh - np.mean(all_soh))**2)
    
    print(f"Test MSE Loss: {test_mse:.6f}")
    print(f"Test MAE Loss: {test_mae:.6f}")
    print(f"Test RMSE Loss: {test_rmse:.6f}")
    print(f"Test R² Score: {r2:.6f}")

    with open("results.txt", "a") as f:
        f.write(f"--- {'Scaling' if use_scaling else 'No Scaling'} ---\n")
        f.write(f"Test MSE Loss: {test_mse:.6f}\n")
        f.write(f"Test MAE Loss: {test_mae:.6f}\n")
        f.write(f"Test RMSE Loss: {test_rmse:.6f}\n")
        f.write(f"Test R² Score: {r2:.6f}\n\n")

    print(f"Generating plots for {'Scaling' if use_scaling else 'No Scaling'}...")
    
    # Load snapshot models for visualization
    snapshot_models = []
    for snap_state in snapshots:
        snap_model = BatteryPINN(n_features=16, use_scaling=use_scaling).to(device)
        snap_model.load_state_dict(snap_state)
        snap_model.eval()
        snapshot_models.append(snap_model)
    
    visualize(data_dir, snapshot_models, plot_dir=plot_dir, holdout_temp=holdout_temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-scaling', action='store_true', help='Disable Arrhenius temporal scaling')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--patience', type=int, default=150, help='Early stopping patience')
    parser.add_argument('--holdout_temp', type=float, default=None, help='Temperature to hold out (e.g. 35.0). If None, random split.')
    parser.add_argument('--plot_dir', type=str, default='plots_scaled', help='Directory to save plots')
    args = parser.parse_args()
    
    train_model('Dataset_1_NCA_battery', epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, use_scaling=not args.no_scaling, holdout_temp=args.holdout_temp, plot_dir=args.plot_dir)
