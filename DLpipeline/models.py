import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_3D(nn.Module):
    '''A class to build 3D CNN (fully-convolutional networks) model architectures on-the-fly
        
        Args::
            convs : A list that specifies (1) number of layers and 
                                          (2) number of conv-channels per layer in the model architecture.
                    Ex: [16, 32, 64] creates 3 layer FCN_3D with 16, 32 and 64 conv. channels resp. followed 
                    by a final convolutional layer to create class predictions.
                    Each layer consists of a block of *Convolution-BatchNorm-ELU*.
                    
            dropout (optional): additionally add a dropout layer before each *Convolution-BatchNorm-ELU* block.
                The value between [0.,1.] represents the amount of 3D dropout to perform.
                the length of this list should be smaller than the length of 'convs' (obviously).
                To add dropout only before the first n layers, give a smaller list of len(dropout) < len(convs). 
            in_shape (optional): The input shape of the images of format (im_x, im_y, im_z)
            out_class (optional): The number of output classes in the classification task
            kernel_size (optional): kernel size to use in the convolutional layers. 
            debug_print (optional): prints shapes at every layer of the conv model for debugging
    '''
    def __init__(self, convs, dropout=[], in_shape=(96, 114, 96), out_classes=2,
                 kernel_size=3,  
                 debug_print=False):
        
        super().__init__()
        
        self.out_classes = out_classes
        self.k = kernel_size
        assert self.k%2==1, "kernel_size must be odd numbers like 3,5,7 for this architecture to dynamically be configured."
        self.debug_print = debug_print
        self.dropout = dropout
        
        # build the convolutional layers
        self.convs = nn.ModuleList([])
        out_shape = np.array(in_shape)        
        for i, (cin, cout) in enumerate(zip([1]+convs, convs)):
            
            layers = []
            # add dropout layer if requested
            if i < len(self.dropout): 
                layers.append(nn.Dropout3d(p=self.dropout[i]))                
            # add 2 convolution layer with provided kernel_size and batch Norm and ELU activations
            # calc number of intermediary channels dynamically
            cmid = cin + (cout-cin)//2
            layers.extend([nn.Conv3d(cin, cmid, kernel_size=self.k, padding=(self.k-1)//2), # full padding
                           nn.BatchNorm3d(cmid),
                           nn.ELU(),
                           nn.Conv3d(cmid, cout, kernel_size=self.k, padding=0), # zero padding
                           nn.BatchNorm3d(cout),
                           nn.ELU()
                          ])            
            self.convs.append(nn.Sequential(*layers))
            # dynamically calculate the output shape of convolutions + pooling (without padding)
            # equation from https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
            out_shape = (out_shape-self.k+1)//2 
            if self.debug_print: out_shape
            assert np.all(out_shape>0), f"out_shape {out_shape} < = 0"
            
        # set the last FCN layer kernel_size such that it produces a single output prediction
        self.finalconv = nn.Conv3d(convs[-1], self.out_classes, kernel_size=out_shape)

        
    def forward(self, t):
        
        # loop over all convolution layers
        for i,conv in enumerate(self.convs):
            
            # 2 X covolution + non-linear Elu operation
            t = conv(t)
            if self.debug_print: print("conv{}>{}".format(i,list(t.shape)))
                
            # perform maxpool with (2x2x2) kernels
            t = F.max_pool3d(t, kernel_size=2, stride=2)
            if self.debug_print: print("pool{}>{}".format(i,list(t.shape)))
                
        # final layer
        t = self.finalconv(t)
        
        # no activations in the last layer since we use cross_entropy_loss with logits
        t = t.reshape(-1, self.out_classes)
        
        if self.debug_print: 
            print("final{}>{}".format(i,list(t.shape)))
            self.debug_print = False        
            
        return t



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
