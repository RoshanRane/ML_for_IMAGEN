import torch
import torch.nn as nn
import torch.nn.functional as F

class SixtyFourNet(nn.Module):
    """A basic CNN network designed to do your first tests. 
    Input MRI images should be of shape [].
    Consists of 5 [Conv-pool-dropout] layers followed by a linear classifier.
    ----------
    drp_rate: The drop out rate used for regulariztion during training.
        
    """
    def __init__(self, drp_rate=0.1, print_size=False):
        """Initialization Process."""
        super().__init__()
        self.drp_rate = drp_rate
        self.print_size = print_size
        self.dropout = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_5 = nn.Conv3d(64, 36, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=4, stride=2, padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(1296, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1)
        ) 
        # NOTE: we need to leave out the last sigmoid activation as the loss function needs logits.

    def encode(self, x):
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_1(x))
        h = self.dropout(self.pool_1(x))
        x = F.elu(self.Conv_2(h))
        if self.print_size: print(x.shape)
        h = self.dropout(self.pool_2(x))
        x = F.elu(self.Conv_3(h))
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_4(x))
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_5(x))
        if self.print_size: print(x.shape)
        h = self.dropout(self.pool_4(x))
        if self.print_size: print(h.shape)        
        return h

    def forward(self, x):
        x = self.encode(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)
    
###############################################################################
class SixtyFourNet2(nn.Module):
    """A basic CNN network designed to do your first tests. 
    Input MRI images should be of shape [].
    Consists of 5 [Conv-pool-dropout] layers followed by a linear classifier.
    ----------
    drp_rate: The drop out rate used for regulariztion during training.
        
    """
    def __init__(self, drp_rate=0.1, print_size=False):
        """Initialization Process."""
        super().__init__()
        self.drp_rate = drp_rate
        self.print_size = print_size
        self.dropout = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(64, 96, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(96, 48, kernel_size=3, stride=1, padding=0)
        self.Conv_5 = nn.Conv3d(48, 36, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=4, stride=2, padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(1296, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1)
        ) 
        # NOTE: we need to leave out the last sigmoid activation as the loss function needs logits.

    def encode(self, x):
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_1(x))
        h = self.dropout(self.pool_1(x))
        x = F.elu(self.Conv_2(h))
        if self.print_size: print(x.shape)
        h = self.dropout(self.pool_2(x))
        x = F.elu(self.Conv_3(h))
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_4(x))
        if self.print_size: print(x.shape)
        x = F.elu(self.Conv_5(x))
        if self.print_size: print(x.shape)
        h = self.dropout(self.pool_4(x))
        if self.print_size: print(h.shape)        
        return h

    def forward(self, x):
        x = self.encode(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)