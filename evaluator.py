import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix
from utils import accuracy, confusion_matrix, to_onehot, FEATURE_COLS, OUTPUT_DIR
from algorithms.knn import KNN
from algorithms.logistic_regression import LogisticRegression
from algorithms.decision_tree import DecisionTree

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_algorithm(model, X_train, y_train, X_val, y_val, X_test, y_test, classes, name, is_index_output=False):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred_val = model.predict(X_val)
    val_time = time.time() - start

    start = time.time()
    y_pred_test = model.predict(X_test)
    test_time = time.time() - start

    if is_index_output:
        idx_to_class = {i: c for i, c in enumerate(classes)}
        y_pred_val = np.array([idx_to_class[p] for p in y_pred_val])
        y_pred_test = np.array([idx_to_class[p] for p in y_pred_test])

    val_acc = accuracy(y_val, y_pred_val)
    test_acc = accuracy(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test, classes)

    return {
        'name': name,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'val_time': val_time,
        'test_time': test_time,
        'y_pred_val': y_pred_val,
        'y_pred_test': y_pred_test,
        'cm': cm,
    }

def run_evaluation(X_train, y_train, X_val, y_val, X_test, y_test, classes):
    results = []

    # Custom KNN
    knn = KNN(k=7, metric='euclidean')
    results.append(evaluate_algorithm(knn, X_train, y_train, X_val, y_val, X_test, y_test, classes, 'Custom KNN (k=7)'))

    # Custom Logistic Regression
    y_train_onehot = to_onehot(y_train, classes)
    lr = LogisticRegression(lr=0.05, epochs=200, batch_size=512)
    results.append(evaluate_algorithm(lr, X_train, y_train_onehot, X_val, y_val, X_test, y_test, classes, 'Custom Logistic Regression', is_index_output=True))

    # Custom Decision Tree
    dt = DecisionTree(max_depth=15, min_samples_split=20)
    results.append(evaluate_algorithm(dt, X_train, y_train, X_val, y_val, X_test, y_test, classes, 'Custom Decision Tree'))

    # Open-source SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    results.append(evaluate_algorithm(svm, X_train, y_train, X_val, y_val, X_test, y_test, classes, 'SVM (sklearn, RBF kernel)'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, res in enumerate(results):
        cm = res['cm']
        im = axes[idx].imshow(cm, cmap='Blues')
        axes[idx].set_title(f"{res['name']}\nTest Acc: {res['test_acc']:.4f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        n = len(classes)
        for i in range(n):
            for j in range(n):
                axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
        axes[idx].set_xticks(range(n))
        axes[idx].set_yticks(range(n))
        axes[idx].set_xticklabels(classes, rotation=45, fontsize=6)
        axes[idx].set_yticklabels(classes, fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/confusion_matrices.png", dpi=150)
    plt.close()

    names = [r['name'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    train_times = [r['train_time'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, val_accs, width, label='Validation Acc', color='skyblue')
    axes[0].bar(x + width/2, test_accs, width, label='Test Acc', color='lightcoral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, fontsize=9)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].legend()
    for i in range(len(names)):
        axes[0].text(i - width/2, val_accs[i] + 0.005, f'{val_accs[i]:.3f}', ha='center', fontsize=8)
        axes[0].text(i + width/2, test_accs[i] + 0.005, f'{test_accs[i]:.3f}', ha='center', fontsize=8)

    axes[1].bar(names, train_times, color='lightgreen')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('Training Time Comparison')
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(train_times):
        axes[1].text(i, v + 0.01, f'{v:.3f}s', ha='center', fontsize=9)

    metrics = ['val_acc', 'test_acc', 'train_time']
    table_data = []
    for r in results:
        table_data.append([r['name'], f"{r['val_acc']:.4f}", f"{r['test_acc']:.4f}", f"{r['train_time']:.3f}s"])
    axes[2].axis('tight')
    axes[2].axis('off')
    table = axes[2].table(cellText=table_data, colLabels=['Algorithm', 'Val Acc', 'Test Acc', 'Train Time'],
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axes[2].set_title('Summary Table')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/comparison.png", dpi=150)
    plt.close()

    if hasattr(lr, 'loss_history') and lr.loss_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lr.loss_history, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title('Logistic Regression Training Loss')
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/figures/loss_curve.png", dpi=150)
        plt.close()

    return results

def robustness_test(X_train, y_train, X_test, y_test, classes, noise_levels=[0.05, 0.1, 0.2]):
    from sklearn.svm import SVC
    from algorithms.knn import KNN
    from algorithms.decision_tree import DecisionTree
    from algorithms.logistic_regression import LogisticRegression
    from utils import to_onehot

    report_lines = []
    report_lines.append("\n" + "=" * 60)
    report_lines.append("ROBUSTNESS ANALYSIS - Noise Injection")
    report_lines.append("=" * 60)

    algorithms = [
        ('KNN (k=7)', lambda: KNN(k=7)),
        ('Logistic Regression', lambda: LogisticRegression(lr=0.05, epochs=100, batch_size=512)),
        ('Decision Tree', lambda: DecisionTree(max_depth=15, min_samples_split=20)),
        ('SVM (RBF)', lambda: SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)),
    ]

    for noise_level in noise_levels:
        X_test_noisy = X_test.copy().astype(float)
        n_features = X_test_noisy.shape[1]
        n_noise = int(noise_level * n_features)
        noise_cols = np.random.choice(n_features, n_noise, replace=False)
        for col in noise_cols:
            std = np.std(X_test_noisy[:, col])
            X_test_noisy[:, col] += np.random.normal(0, std * noise_level, size=X_test_noisy.shape[0])

        report_lines.append(f"\n--- Noise Level: {noise_level*100:.0f}% features with std*{noise_level} noise ---")
        for alg_name, alg_factory in algorithms:
            try:
                model = alg_factory()
                if alg_name == 'Logistic Regression':
                    y_train_onehot = to_onehot(y_train, classes)
                    model.fit(X_train, y_train_onehot)
                else:
                    model.fit(X_train, y_train)
                y_pred = model.predict(X_test_noisy)
                if alg_name == 'Logistic Regression':
                    idx_to_class = {i: c for i, c in enumerate(classes)}
                    y_pred = np.array([idx_to_class[p] for p in y_pred])
                acc = accuracy(y_test, y_pred)
                report_lines.append(f"  {alg_name:25s}: Test Acc = {acc:.4f}")
            except Exception as e:
                report_lines.append(f"  {alg_name:25s}: Error - {str(e)[:50]}")

    return "\n".join(report_lines)

def overfitting_analysis(results, X_train, y_train, X_val, y_val, classes):
    report_lines = []
    report_lines.append("\n" + "=" * 60)
    report_lines.append("OVERFITTING ANALYSIS")
    report_lines.append("=" * 60)
    for r in results:
        diff = abs(r['val_acc'] - r['test_acc'])
        report_lines.append(f"\n{r['name']}:")
        report_lines.append(f"  Validation Accuracy: {r['val_acc']:.4f}")
        report_lines.append(f"  Test Accuracy:       {r['test_acc']:.4f}")
        report_lines.append(f"  Gap:                 {diff:.4f}")
        if diff < 0.02:
            report_lines.append(f"  Verdict: No significant overfitting")
        elif diff < 0.05:
            report_lines.append(f"  Verdict: Mild overfitting")
        else:
            report_lines.append(f"  Verdict: Significant overfitting - consider regularization")
    return "\n".join(report_lines)
