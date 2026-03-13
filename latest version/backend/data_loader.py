import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class GenericDataset(Dataset):
    def __init__(self, X, y, priv_mask, unpriv_mask):
        self.X = X
        self.y = y
        self.priv_mask = priv_mask
        self.unpriv_mask = unpriv_mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.priv_mask[idx], self.unpriv_mask[idx]

def get_adult_dataloaders(batch_size=256):
    logger.info("Loading Adult dataset...")
    data = fetch_adult(as_frame=True)
    X, y = data.data, (data.target == '>50K') * 1
    sex = X['sex']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.select_dtypes(include=['category', 'object']).columns)
    ])
    
    X_p = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X_p, y, sex, test_size=0.3, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    priv_mask_train = torch.tensor((s_train == 'Male').values, dtype=torch.bool)
    unpriv_mask_train = torch.tensor((s_train == 'Female').values, dtype=torch.bool)
    
    train_dataset = GenericDataset(X_train_t, y_train_t, priv_mask_train, unpriv_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_data = {
        'X_test_t': torch.tensor(X_test, dtype=torch.float32),
        'y_test': y_test.values,
        'sex_test': s_test
    }
    
    return train_loader, test_data, X_train.shape[1]

def get_synthetic_vision_dataloaders(batch_size=64):
    logger.info("Generating synthetic vision dataset for testing...")
    # Generate fake 3x64x64 images
    num_train = 2000
    num_test = 500
    
    X_train = torch.randn(num_train, 3, 64, 64)
    y_train = torch.randint(0, 2, (num_train, 1)).float()
    sens_train = torch.randint(0, 2, (num_train,))
    priv_mask_train = sens_train == 1
    unpriv_mask_train = sens_train == 0
    
    train_dataset = GenericDataset(X_train, y_train, priv_mask_train, unpriv_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_test = torch.randn(num_test, 3, 64, 64)
    y_test = torch.randint(0, 2, (num_test, 1)).float()
    sens_test = torch.randint(0, 2, (num_test,))
    
    # Map the synthetic sensitive attribute back to "Male" / "Female" for Fairlearn DPD calculation
    sex_test_mapped = pd.Series(["Male" if val == 1 else "Female" for val in sens_test.numpy()])
    
    test_data = {
        'X_test_t': X_test,
        'y_test': y_test.numpy().flatten(),
        'sex_test': sex_test_mapped
    }
    
    return train_loader, test_data, None
