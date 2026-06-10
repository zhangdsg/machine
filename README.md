# Dry Bean Dataset - Machine Learning Project

## Project Overview
This project implements a complete machine learning pipeline for classifying 7 types of dry beans using the Dry Bean Dataset. The dataset has been artificially contaminated with missing values and label errors for data cleaning practice.

## Dataset
- **Source**: Dry Bean Dataset (UCI)
- **Samples**: 13,611 (train: 9527, validation: 1347, test: 2737)
- **Features**: 16 geometric features (Area, Perimeter, MajorAxisLength, etc.)
- **Classes**: 7 bean types (DERMASON, BARBUNYA, HOROZ, SEKER, SIRA, CALI, BOMBAY)

## Project Structure
```
BeanClassifier/
├── main.py                  # Unified entry point
├── data_loader.py           # Data loading module
├── data_cleaner.py          # Data cleaning (missing values, label normalization)
├── eda.py                   # Exploratory data analysis & visualization
├── evaluator.py             # Model evaluation, comparison, robustness testing
├── utils.py                 # Common utilities (standardization, metrics)
├── algorithms/
│   ├── knn.py               # Custom K-Nearest Neighbors
│   ├── logistic_regression.py  # Custom Logistic Regression (Softmax)
│   └── decision_tree.py     # Custom Decision Tree (CART)
├── output/
│   ├── report.txt           # Full analysis report
│   └── figures/             # Generated visualizations
└── README.md
```

## Implemented Algorithms

### Custom Implementations (from scratch)
1. **K-Nearest Neighbors** - Euclidean distance, k=7
2. **Logistic Regression** - Softmax, mini-batch gradient descent
3. **Decision Tree** - CART algorithm, Gini impurity

### Open-Source Library
4. **SVM** (scikit-learn) - RBF kernel, C=1.0

## Results

| Algorithm | Val Acc | Test Acc | Train Time |
|-----------|---------|----------|------------|
| KNN (k=7) | 0.9183 | 0.9214 | 0.000s |
| Logistic Regression | 0.9220 | 0.9167 | 0.667s |
| Decision Tree | 0.9072 | 0.9024 | 1.129s |
| SVM (RBF) | 0.9295 | 0.9288 | 0.209s |

## How to Run
```bash
python main.py
```

All outputs (report, figures) will be saved to the `output/` directory.

## Dependencies
- Python 3.8+
- numpy, pandas, matplotlib, seaborn, scikit-learn
