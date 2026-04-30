import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
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

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 将图像展平为向量
def flatten_dataset(dataset):
    X = []
    y = []
    for img, label in dataset:
        X.append(img.view(-1).numpy())
        y.append(label)
    return np.array(X), np.array(y)

X_train_full, y_train_full = flatten_dataset(trainset)
X_test_full, y_test_full = flatten_dataset(testset)

# 划分验证集（从训练集中取20%）
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test_full)
joblib.dump(scaler, 'models/cifar10_scaler.pkl')

# 转换为张量
train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train).long())
val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val).long())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 模型
input_dim = X_train.shape[1]  # 32*32*3 = 3072
model = ANN(input_dim, hidden_dims=[512, 256, 128], output_dim=10, task='multiclass_classification').to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30  # 30个epoch已经较慢，可以适当减少
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
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = correct / total
    val_accs.append(val_acc)

    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

torch.save(model.state_dict(), 'models/ann_cifar10.pth')

# 绘制曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CIFAR-10 Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 Accuracy')
plt.legend()
plt.savefig('results/cifar10_ann_curves.png')
plt.close()

print(f'Final Validation Accuracy: {val_accs[-1]:.4f}')
with open('results/cifar10_ann_accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {val_accs[-1]}\n')