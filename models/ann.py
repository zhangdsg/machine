import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, task='classification'):
        """
        input_dim: 输入特征维度
        hidden_dims: list of int, 例如 [128, 64]
        output_dim: 输出维度（回归=1，二分类=1，多分类=类别数）
        task: 'regression', 'binary_classification', 'multiclass_classification'
        """
        super(ANN, self).__init__()
        self.task = task
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.task == 'binary_classification':
            return torch.sigmoid(out)
        elif self.task == 'multiclass_classification':
            return F.log_softmax(out, dim=1)
        else:  # regression
            return out