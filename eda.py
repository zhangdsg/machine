import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import FEATURE_COLS, OUTPUT_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_eda(train_df, val_df, test_df):
    dfs = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
    report_lines.append("=" * 60)

    for name, df in dfs.items():
        report_lines.append(f"\n--- {name} Set ---")
        report_lines.append(f"  Shape: {df.shape}")
        numeric_cols = df[FEATURE_COLS].select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe().to_string()
        report_lines.append(f"  Feature statistics:\n{stats}")

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        for name, df in dfs.items():
            if col in df.select_dtypes(include=[np.number]).columns:
                sns.kdeplot(df[col].dropna(), label=name, ax=axes[i], fill=True, alpha=0.3)
        axes[i].set_title(f'{col} Distribution')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/feature_distributions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        for name, df in dfs.items():
            if col in df.select_dtypes(include=[np.number]).columns:
                sns.boxplot(x='Class', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} by Class')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/feature_by_class.png", dpi=150)
    plt.close()

    numeric_df = train_df[FEATURE_COLS].select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/correlation_matrix.png", dpi=150)
    plt.close()

    class_counts = train_df['Class'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    ax.bar(class_counts.index, class_counts.values, color=colors)
    ax.set_title('Class Distribution in Training Set')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/class_distribution.png", dpi=150)
    plt.close()

    report_lines.append(f"\nClass distribution after cleaning:")
    for cls, cnt in class_counts.items():
        report_lines.append(f"  {cls}: {cnt}")

    return "\n".join(report_lines)
