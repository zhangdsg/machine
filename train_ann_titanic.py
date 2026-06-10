import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
from models.ann import ANN
from preprocess.titanic_preprocess import load_and_clean_titanic

os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
X_full, y_full, _ = load_and_clean_titanic('data/titanic_train_knn.csv', 'data/titanic_test_knn.csv')
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, 'models/titanic_scaler.pkl')

# 转换为张量
train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), 
                              torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)), 
                            torch.tensor(y_val.values.astype(np.float32)).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型
input_dim = X_train.shape[1]
model = ANN(input_dim, hidden_dims=[64, 32], output_dim=1, task='binary_classification').to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 150
train_losses = []
val_losses = []
val_accs = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = correct / total
    val_accs.append(val_acc)

    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

torch.save(model.state_dict(), 'models/ann_titanic.pth')

# 绘制曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Titanic Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Titanic Accuracy')
plt.legend()
plt.savefig('results/titanic_ann_curves.png')
plt.close()

print(f'Final Validation Accuracy: {val_accs[-1]:.4f}')
with open('results/titanic_ann_accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {val_accs[-1]}\n')