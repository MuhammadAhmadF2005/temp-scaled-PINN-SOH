import torch
import numpy as np
import argparse
from data_loader import get_dataloaders
from model import BatteryPINN, compute_pinn_losses
from visualize import visualize
def train_model(data_dir, epochs=500, batch_size=128, patience=100, use_scaling=True, holdout_temp=None, plot_dir='plots_scaled'):
    print("Loading data...")
    train_loader, val_loader, test_loader, soh_min, soh_range = get_dataloaders(data_dir, holdout_temp=holdout_temp, batch_size=batch_size, max_train_cycles=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Initializing model (use_scaling={use_scaling})...")
    model = BatteryPINN(n_features=16, use_scaling=use_scaling).to(device)
    
    # Differential learning rates: Ea gets 1e-2, others get 1e-3
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
        train_mse = 0.0
        train_pde = 0.0
        train_mono = 0.0
        
        for batch_i, (features, soh, temp, cycle) in enumerate(train_loader):
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            features.requires_grad_(True)
            optimizer.zero_grad()
            
            loss, mse_loss, pde_loss, mono_loss = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.01, beta=0.1)
            
            loss.backward()
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
        
        scheduler.step(val_loss)
        
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
                
    print("Training finished. Evaluating on test set...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    test_mse = 0.0
    test_mae = 0.0
    all_u = []
    all_soh = []
    
    with torch.no_grad():
        for features, soh, temp, cycle in test_loader:
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            u, _ = model(features, temp, cycle)
            
            mse = torch.nn.MSELoss()(u, soh).item()
            mae = torch.nn.L1Loss()(u, soh).item()
            
            test_mse += mse * features.size(0)
            test_mae += mae * features.size(0)
            
            all_u.append(u.cpu().numpy())
            all_soh.append(soh.cpu().numpy())
            
    all_u = np.concatenate(all_u) * soh_range + soh_min
    all_soh = np.concatenate(all_soh) * soh_range + soh_min
    
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
    visualize(data_dir, model, plot_dir=plot_dir, holdout_temp=holdout_temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-scaling', action='store_true', help='Disable Arrhenius temporal scaling')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--holdout_temp', type=float, default=None, help='Temperature to hold out (e.g. 35.0). If None, random split.')
    parser.add_argument('--plot_dir', type=str, default='plots_scaled', help='Directory to save plots')
    args = parser.parse_args()
    
    train_model('Dataset_1_NCA_battery', epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, use_scaling=not args.no_scaling, holdout_temp=args.holdout_temp, plot_dir=args.plot_dir)
