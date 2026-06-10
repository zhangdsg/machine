import sys
import numpy as np
import pandas as pd
from data_loader import load_datasets
from data_cleaner import clean_datasets, report_cleaning
from eda import run_eda
from evaluator import run_evaluation, robustness_test, overfitting_analysis
from utils import FEATURE_COLS, OUTPUT_DIR, standardize, accuracy

def main():
    print("=" * 60)
    print("Dry Bean Dataset - Machine Learning Project")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading datasets...")
    train_raw, val_raw, test_raw = load_datasets()
    print(f"  Train: {train_raw.shape}, Val: {val_raw.shape}, Test: {test_raw.shape}")

    # Step 2: Data analysis (EDA)
    print("\n[2/6] Exploratory Data Analysis...")
    eda_report = run_eda(train_raw, val_raw, test_raw)
    print(eda_report)

    # Step 3: Data cleaning
    print("\n[3/6] Data Cleaning...")
    train, val, test = clean_datasets(train_raw, val_raw, test_raw)
    cleaning_report = report_cleaning(train_raw, train)
    print(cleaning_report)

    # Step 4: Prepare features
    print("\n[4/6] Feature Preparation...")
    X_train_raw = train[FEATURE_COLS].values.astype(float)
    X_val_raw = val[FEATURE_COLS].values.astype(float)
    X_test_raw = test[FEATURE_COLS].values.astype(float)

    X_train, X_val, X_test, mean, std = standardize(X_train_raw, X_val_raw, X_test_raw)

    classes = sorted(train['Class'].unique())
    y_train = train['Class'].values
    y_val = val['Class'].values
    y_test = test['Class'].values

    for label, arr in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        print(f"  {label}: {arr.shape[0]} samples, {arr.shape[1]} features")
    print(f"  Classes: {classes}")

    # Step 5: Algorithm implementation and evaluation
    print("\n[5/6] Algorithm Implementation and Evaluation...")
    results = run_evaluation(X_train, y_train, X_val, y_val, X_test, y_test, classes)

    # Loss curve for logistic regression
    from algorithms.logistic_regression import LogisticRegression
    from utils import to_onehot
    lr_full = LogisticRegression(lr=0.05, epochs=200, batch_size=512)
    y_train_oh = to_onehot(y_train, classes)
    lr_full.fit(X_train, y_train_oh)

    # Save all results
    print("\n[6/6] Generating Reports...")
    combined_report = []
    combined_report.append("=" * 70)
    combined_report.append("DRY BEAN DATASET - MACHINE LEARNING PROJECT REPORT")
    combined_report.append("=" * 70)

    combined_report.append("\n\n--- PART 1: DATA INTRODUCTION ---")
    combined_report.append("""
The Dry Bean Dataset contains 13,611 samples of 7 different types of dry beans
(DERMASON, BARBUNYA, HOROZ, SEKER, SIRA, CALI, BOMBAY).
Each sample has 16 geometric features extracted from bean images:
- Area, Perimeter, MajorAxisLength, MinorAxisLength
- AspectRation, Eccentricity, ConvexArea, EquivDiameter
- Extent, Solidity, roundness, Compactness
- ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

The dataset has been pre-split into train (9527), validation (1347), and test (2737) sets,
with artificial contamination (missing values, label errors) for cleaning practice.
""")

    combined_report.append(eda_report)
    combined_report.append(cleaning_report)

    combined_report.append("\n\n--- PART 4: ALGORITHM COMPARISON ---")
    combined_report.append(f"{'Algorithm':<30} {'Val Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    combined_report.append("-" * 66)
    for r in results:
        combined_report.append(f"{r['name']:<30} {r['val_acc']:<12.4f} {r['test_acc']:<12.4f} {r['train_time']:<12.3f}s")

    combined_report.append("\n\n--- PART 5: OVERFITTING ANALYSIS ---")
    oa = overfitting_analysis(results, X_train, y_train, X_val, y_val, classes)
    combined_report.append(oa)

    combined_report.append("\n\n--- PART 6: ROBUSTNESS ANALYSIS ---")
    rt = robustness_test(X_train, y_train, X_test, y_test, classes)
    combined_report.append(rt)

    combined_report.append("\n\n--- PART 7: COURSE SUMMARY ---")
    combined_report.append("""
Through this machine learning course and project, I have learned:
1. The complete ML pipeline: data collection, preprocessing, feature engineering, model training, evaluation
2. How to handle real-world data issues: missing values, label noise, outliers
3. Implementation of classic algorithms from scratch: KNN, Logistic Regression, Decision Tree
4. Using open-source libraries (scikit-learn) for efficient model development (SVM)
5. Model evaluation techniques: accuracy, confusion matrix, overfitting analysis, robustness testing
6. Project structuring and version control with Git/GitHub
7. The importance of data quality and proper preprocessing for model performance

Course suggestions:
- More hands-on projects with real-world datasets
- More time for advanced topics like ensemble methods and neural networks
- Include a session on ML system design and deployment
""")

    final_report = "\n".join(combined_report)
    with open(f"{OUTPUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
