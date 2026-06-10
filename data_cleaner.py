import pandas as pd
import numpy as np
from utils import clean_dataframe, FEATURE_COLS, CLASS_MAPPING

def clean_datasets(train, val, test):
    train_clean = clean_dataframe(train)
    val_clean = clean_dataframe(val)
    test_clean = clean_dataframe(test)
    return train_clean, val_clean, test_clean

def report_cleaning(before_train, after_train):
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATA CLEANING REPORT")
    report_lines.append("=" * 60)

    report_lines.append(f"\nOriginal training samples: {len(before_train)}")
    report_lines.append(f"Cleaned training samples: {len(after_train)}")

    missing_before = before_train[['Perimeter', 'Solidity', 'Compactness']].isnull().sum()
    report_lines.append(f"\nMissing values before cleaning (train):")
    for col, cnt in missing_before.items():
        report_lines.append(f"  {col}: {cnt}")

    report_lines.append(f"\nUnique classes before: {sorted(before_train['Class'].unique())}")
    report_lines.append(f"Unique classes after: {sorted(after_train['Class'].unique())}")

    before_counts = before_train['Class'].value_counts()
    after_counts = after_train['Class'].value_counts()
    report_lines.append(f"\nClass distribution before (top 10):")
    for cls, cnt in before_counts.head(10).items():
        report_lines.append(f"  '{cls}': {cnt}")
    report_lines.append(f"\nClass distribution after:")
    for cls, cnt in after_counts.items():
        report_lines.append(f"  {cls}: {cnt}")

    return "\n".join(report_lines)
