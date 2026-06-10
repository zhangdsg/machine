import torch
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from models.ann import ANN
from preprocess.titanic_preprocess import load_and_clean_titanic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和 scaler
model = ANN(input_dim=7, hidden_dims=[64, 32], output_dim=1, task='binary_classification').to(device)
model.load_state_dict(torch.load('models/ann_titanic.pth', map_location=device))
model.eval()
scaler = joblib.load('models/titanic_scaler.pkl')

# 准备测试数据（从训练集划分一部分作为测试集）
X_full, y_full, _ = load_and_clean_titanic('data/titanic_train_knn.csv', 'data/titanic_test_knn.csv')
_, X_test, _, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = (outputs > 0.5).float().cpu().numpy().flatten()
acc = (preds == y_test.values).mean()
print(f'Test Accuracy: {acc:.4f}')
with open('results/titanic_ann_test_accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')