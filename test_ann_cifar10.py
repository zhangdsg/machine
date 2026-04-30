import torch
import joblib
import numpy as np
import torchvision
import torchvision.transforms as transforms
from models.ann import ANN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和 scaler
model = ANN(input_dim=3072, hidden_dims=[512, 256, 128], output_dim=10, task='multiclass_classification').to(device)
model.load_state_dict(torch.load('models/ann_cifar10.pth', map_location=device))
model.eval()
scaler = joblib.load('models/cifar10_scaler.pkl')

# 加载测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 展平并标准化
X_test = []
y_test = []
for img, label in testset:
    X_test.append(img.view(-1).numpy())
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = scaler.transform(X_test)

# 预测
X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
batch_size = 128
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False)
preds = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch[0])
        preds.append(outputs.argmax(dim=1).cpu().numpy())
preds = np.concatenate(preds)
acc = (preds == y_test).mean()
print(f'Test Accuracy: {acc:.4f}')
with open('results/cifar10_ann_test_accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')