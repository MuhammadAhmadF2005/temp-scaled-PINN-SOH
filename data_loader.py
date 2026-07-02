import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis, skew, entropy

class BatterySOHDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def extract_temperature(filename):
    match = re.search(r'CY(\d+)', filename)
    if match:
        return float(match.group(1))
    return 25.0

def compute_features(df_cycle):
    # Focus on the charging phase: <I>/mA > 0 (or some small positive threshold)
    df_charge = df_cycle[df_cycle['<I>/mA'] > 10.0].copy()
    if df_charge.empty:
        return np.zeros(16)
        
    v_arr = df_charge['Ecell/V'].values
    i_arr = df_charge['<I>/mA'].values
    t_arr = df_charge['time/s'].values
    q_arr = df_charge['Q charge/mA.h'].values
    
    v_end = np.max(v_arr) if len(v_arr) > 0 else 4.2
    
    # Segment 1: CC mode voltage features [V_end - 0.2, V_end]
    v_mask = (v_arr >= v_end - 0.2) & (v_arr <= v_end)
    seg_v = v_arr[v_mask]
    seg_t_v = t_arr[v_mask]
    seg_q_v = q_arr[v_mask]
    
    # Segment 2: CV mode current features [500mA, 100mA]
    # CV mode starts approximately when voltage peaks
    idx_vmax = np.argmax(v_arr)
    cv_v_arr = v_arr[idx_vmax:]
    cv_i_arr = i_arr[idx_vmax:]
    cv_t_arr = t_arr[idx_vmax:]
    cv_q_arr = q_arr[idx_vmax:]
    
    i_mask = (cv_i_arr <= 500.0) & (cv_i_arr >= 100.0)
    seg_i = cv_i_arr[i_mask]
    seg_t_i = cv_t_arr[i_mask]
    seg_q_i = cv_q_arr[i_mask]
    
    def extract_8_features(sig, t_seg, q_seg):
        if len(sig) < 2:
            return [0.0] * 8
        m = np.mean(sig)
        s = np.std(sig)
        k = kurtosis(sig)
        sk = skew(sig)
        chg_time = np.max(t_seg) - np.min(t_seg)
        acc_q = np.max(q_seg) - np.min(q_seg)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            slope = np.polyfit(t_seg, sig, 1)[0] if len(t_seg) > 1 else 0.0
        
        hist, _ = np.histogram(sig, bins=10, density=True)
        hist = hist[hist > 0]
        ent = entropy(hist) if len(hist) > 0 else 0.0
        
        return [m, s, k, sk, chg_time, acc_q, slope, ent]
        
    f1 = extract_8_features(seg_v, seg_t_v, seg_q_v)
    f2 = extract_8_features(seg_i, seg_t_i, seg_q_i)
    
    features = f1 + f2
    return np.nan_to_num(np.array(features))

def prepare_dataset(data_dir, holdout_temp=None, val_ratio=0.1, test_ratio=0.2, max_train_cycles=100):
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    train_val_files = []
    test_files = []
    
    if holdout_temp is not None:
        for f in all_files:
            temp = extract_temperature(os.path.basename(f))
            if temp == holdout_temp:
                test_files.append(f)
            else:
                train_val_files.append(f)
                
        np.random.seed(42)
        np.random.shuffle(train_val_files)
        
        n_val = max(1, int(len(train_val_files) * val_ratio))
        val_files = train_val_files[:n_val]
        train_files = train_val_files[n_val:]
    else:
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        n_test = max(1, int(len(all_files) * test_ratio))
        n_val = max(1, int(len(all_files) * val_ratio))
        
        test_files = all_files[:n_test]
        val_files = all_files[n_test:n_test+n_val]
        train_files = all_files[n_test+n_val:]
    
    def process_files(files, is_training=False):
        data_list = []
        for file in files:
            temp = extract_temperature(os.path.basename(file))
            df = pd.read_csv(file)
            
            cycles = df['cycle number'].unique()
            baseline_soh = None
            
            # Step 1: Collect raw valid cycles
            valid_raw_cycles = []
            for cycle_num in sorted(cycles):
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
                
                valid_raw_cycles.append({
                    'df_c': df_c,
                    'soh': soh
                })
            
            if not valid_raw_cycles:
                continue
                
            # Step 2: Smooth valid SOH values using a rolling median filter
            soh_raw_vals = [c['soh'] for c in valid_raw_cycles]
            s_series = pd.Series(soh_raw_vals)
            soh_smoothed = s_series.rolling(window=5, center=True, min_periods=1).median().values
            
            # Step 3: Compute features and build data list, applying max_train_cycles if training
            for idx, c in enumerate(valid_raw_cycles):
                if is_training and max_train_cycles is not None and idx >= max_train_cycles:
                    break
                features = compute_features(c['df_c'])
                data_list.append({
                    'features': features,
                    'soh': soh_smoothed[idx],
                    'temperature': temp,
                    'cycle': idx + 1
                })
        return data_list

    print(f"Processing training data ({len(train_files)} files)...")
    train_data = process_files(train_files, is_training=True)
    print(f"Processing validation data ({len(val_files)} files)...")
    val_data = process_files(val_files, is_training=True)
    if holdout_temp is not None:
        print(f"Processing testing data ({len(test_files)} files) at {holdout_temp} °C...")
    else:
        print(f"Processing testing data ({len(test_files)} files) from mixed temperatures...")
    test_data = process_files(test_files, is_training=False)
    
    all_features = np.array([d['features'] for d in train_data])
    f_min = all_features.min(axis=0, keepdims=True)
    f_max = all_features.max(axis=0, keepdims=True)
    
    f_range = f_max - f_min
    f_range[f_range == 0] = 1e-6
    
    all_soh = np.array([d['soh'] for d in train_data])
    soh_min = all_soh.min()
    soh_max = all_soh.max()
    soh_range = soh_max - soh_min
    if soh_range == 0: soh_range = 1e-6
    
    def normalize_list(d_list):
        for d in d_list:
            feat = d['features']
            feat_norm = 2 * ((feat - f_min) / f_range) - 1.0
            d['features'] = feat_norm.squeeze()
            
            d['soh_norm'] = (d['soh'] - soh_min) / soh_range
        return d_list
        
    train_data = normalize_list(train_data)
    val_data = normalize_list(val_data)
    test_data = normalize_list(test_data)
    
    return train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range

def get_dataloaders(data_dir, holdout_temp=None, batch_size=64, max_train_cycles=100):
    train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range = prepare_dataset(data_dir, holdout_temp=holdout_temp, max_train_cycles=max_train_cycles)
    
    def collate_fn(batch):
        features = torch.tensor(np.array([item['features'] for item in batch]), dtype=torch.float32)
        soh = torch.tensor(np.array([item['soh_norm'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        temp = torch.tensor(np.array([item['temperature'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        cycle = torch.tensor(np.array([item['cycle'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        return features, soh, temp, cycle

    train_loader = DataLoader(BatterySOHDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BatterySOHDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(BatterySOHDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, soh_min, soh_range
