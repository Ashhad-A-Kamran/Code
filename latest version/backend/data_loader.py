import os
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

# Path to local COMPASS dataset
_COMPASS_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "datasets", "COMPASS",
        "propublicaCompassRecividism_data_fairml.csv",
        "propublica_data_for_fairml.csv",
    )
)


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


# ---------------------------------------------------------------------------
# Adult Income Dataset
# ---------------------------------------------------------------------------
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
    X_p_df = pd.DataFrame(X_p, columns=preprocessor.get_feature_names_out())

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_p_df, y, sex, test_size=0.3, random_state=42
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    priv_mask_train = torch.tensor((s_train == 'Male').values, dtype=torch.bool)
    unpriv_mask_train = torch.tensor((s_train == 'Female').values, dtype=torch.bool)

    train_dataset = GenericDataset(X_train_t, y_train_t, priv_mask_train, unpriv_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = {
        'X_test_t': torch.tensor(X_test.values, dtype=torch.float32),
        'X_test_df': X_test,
        'y_test': y_test.values,
        'sex_test': s_test,
    }

    return train_loader, test_data, X_train.shape[1]


# ---------------------------------------------------------------------------
# COMPASS (ProPublica Recidivism) Dataset
# ---------------------------------------------------------------------------
def get_compass_dataloaders(batch_size=256):
    logger.info("Loading COMPASS dataset...")

    df = pd.read_csv(_COMPASS_CSV)
    # Columns: Two_yr_Recidivism, Number_of_Priors, score_factor,
    #          Age_Above_FourtyFive, Age_Below_TwentyFive,
    #          African_American, Asian, Hispanic, Native_American, Other, Female, Misdemeanor

    target_col = 'Two_yr_Recidivism'
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Sensitive: Female (0=Male/privileged, 1=Female/unprivileged)
    sex = X['Female'].map({0: 'Male', 1: 'Female'})

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_scaled, y, sex, test_size=0.3, random_state=42
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    priv_mask_train = torch.tensor((s_train == 'Male').values, dtype=torch.bool)
    unpriv_mask_train = torch.tensor((s_train == 'Female').values, dtype=torch.bool)

    train_dataset = GenericDataset(X_train_t, y_train_t, priv_mask_train, unpriv_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = {
        'X_test_t': torch.tensor(X_test.values, dtype=torch.float32),
        'X_test_df': X_test,
        'y_test': y_test.values,
        'sex_test': s_test,
    }

    return train_loader, test_data, X_train.shape[1]


# ---------------------------------------------------------------------------
# German Credit Dataset (from UCI)
# ---------------------------------------------------------------------------
def get_german_credit_dataloaders(batch_size=64):
    logger.info("Loading German Credit dataset...")

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    feature_names = [
        'status', 'duration', 'credit_history', 'purpose', 'amount',
        'savings', 'employment_duration', 'installment_rate', 'statussex',
        'other_debtors', 'residence_since', 'property', 'age',
        'other_installment_plans', 'housing', 'number_credits', 'job',
        'people_liable', 'telephone', 'foreign_worker', 'credit_risk'
    ]
    data = pd.read_csv(url, header=None, sep=' ', names=feature_names)

    categorical_cols = [
        'status', 'credit_history', 'purpose', 'savings', 'employment_duration',
        'statussex', 'other_debtors', 'property', 'other_installment_plans',
        'housing', 'job', 'telephone', 'foreign_worker'
    ]
    data = pd.get_dummies(data, columns=categorical_cols)
    data['credit_risk'] = data['credit_risk'].replace({1: 1, 2: 0})

    target_col = 'credit_risk'
    y = data[target_col]
    X = data.drop(columns=[target_col])

    # Sensitive: statussex_A92 = Female (unprivileged)
    sex_col = 'statussex_A92'
    if sex_col in X.columns:
        sex = X[sex_col].map({1: 'Female', 0: 'Male'})
    else:
        sex = pd.Series(['Male'] * len(X), index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_scaled, y, sex, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    priv_mask_train = torch.tensor((s_train == 'Male').values, dtype=torch.bool)
    unpriv_mask_train = torch.tensor((s_train == 'Female').values, dtype=torch.bool)

    train_dataset = GenericDataset(X_train_t, y_train_t, priv_mask_train, unpriv_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = {
        'X_test_t': torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32),
        'X_test_df': X_test,
        'y_test': y_test.values,
        'sex_test': s_test,
    }

    return train_loader, test_data, X_train.shape[1]


# ---------------------------------------------------------------------------
# Synthetic Vision (ResNet testing)
# ---------------------------------------------------------------------------
def get_synthetic_vision_dataloaders(batch_size=64):
    logger.info("Generating synthetic vision dataset for testing...")
    num_train, num_test = 2000, 500

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

    sex_test_mapped = pd.Series(["Male" if val == 1 else "Female" for val in sens_test.numpy()])

    test_data = {
        'X_test_t': X_test,
        'X_test_df': None,
        'y_test': y_test.numpy().flatten(),
        'sex_test': sex_test_mapped,
    }

    return train_loader, test_data, None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def get_dataloaders(dataset_name: str, batch_size: int = 256):
    """Return (train_loader, test_data, n_features) for the given dataset name."""
    loaders = {
        "adult":         get_adult_dataloaders,
        "compass":       get_compass_dataloaders,
        "german_credit": get_german_credit_dataloaders,
    }
    fn = loaders.get(dataset_name)
    if fn is None:
        logger.warning(f"Unknown dataset '{dataset_name}', defaulting to Adult.")
        fn = get_adult_dataloaders
    return fn(batch_size)
