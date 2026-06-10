import pandas as pd
from utils import DATA_DIR

def load_datasets():
    train = pd.read_csv(f"{DATA_DIR}/Dry_Bean_Dataset_Dirty_train.csv")
    val = pd.read_csv(f"{DATA_DIR}/Dry_Bean_Dataset_Dirty_val.csv")
    test = pd.read_csv(f"{DATA_DIR}/Dry_Bean_Dataset_Dirty_test.csv")
    return train, val, test
