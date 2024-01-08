from torch import nn
import torch.nn.functional as F
from torch import Tensor

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x: Tensor):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.log_softmax(x, dim=1)
        
        return x

'''
import torch

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        self.l1 = torch.nn.Linear(in_features, 500)
        self.l2 = torch.nn.Linear(500, out_features)
        self.r = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))
'''
