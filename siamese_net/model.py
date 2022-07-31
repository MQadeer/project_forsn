import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
    Author: Alan Preciado
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                     )

        self.fc = nn.Sequential(nn.Linear(30976, 64),
                                nn.PReLU(),
                                nn.Linear(64, 64),
                                nn.BatchNorm1d(64),
                                nn.PReLU(),
                                nn.Linear(64, 50))

    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
