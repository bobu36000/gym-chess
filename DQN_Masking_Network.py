
# PyTorch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class for a convolutional neural network
class Network(nn.Module):
    def __init__(self, lr, channels, layer_dims, kernel_size, stride, reduction=None):
        super().__init__()
        self.input_channels = 12
        self.output_dim = 4100 

        # define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels, out_channels=channels[0], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, stride=stride),
            nn.ReLU()
        )

        # calculate the input size to the fully connected layer
        self.fc_input_size = self._get_fc_input_size()

        # define fully connected linear layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, layer_dims[0]),
            nn.ReLU(),
            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]),
            nn.ReLU(),
            nn.Linear(layer_dims[2], self.output_dim)
        )

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if reduction is not None:
            self.loss = nn.HuberLoss(reduction=reduction)
        else:
            self.loss = nn.HuberLoss()

    def _get_fc_input_size(self):
        # Initialize a fake input tensor to compute the output size of conv_layers
        x = T.zeros((1, self.input_channels, 8, 8))
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # x is a torch tensor with dimensions (batch_size, channels, height, width) e.g (1, 1, 8, 8)
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_size)
        x = self.fc_layers(x)
        return x