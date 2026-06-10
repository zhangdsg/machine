import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
from models.ann import ANN

os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
data = pd.read_csv('data/house_data.csv')
X = data[['x1', 'x2', 'x3', 'x4']].values.astype(np.float32)
y = data['y'].values.astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, 'models/house_scaler.pkl')

# 转换为张量
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型
input_dim = X_train.shape[1]
model = ANN(input_dim, hidden_dims=[64, 32], output_dim=1, task='regression').to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
train_losses = []
val_losses = []

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
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'models/ann_house.pth')

# 绘制损失曲线
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('House Price Regression Loss')
plt.legend()
plt.savefig('results/house_loss_curve.png')
plt.close()

# 计算最终验证 MSE
model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val).to(device)
    y_pred = model(X_val_tensor).cpu().numpy()
    mse = np.mean((y_pred - y_val)**2)
print(f'Final Validation MSE: {mse:.2f}')
with open('results/house_ann_mse.txt', 'w') as f:
    f.write(f'MSE: {mse}\n')