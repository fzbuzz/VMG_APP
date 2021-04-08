import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.blocks(x)

class BioFace_FCN(nn.Module):
    def __init__(self, nfilters, LightVectorSize=15, bSize=2):
        super(BioFace_FCN, self).__init__()
        # Some processing
        down_filters = [3] + nfilters #filters, we start with 3 for 3 color channel rgb
        up_filters = nfilters[::-1]

        #DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for in_channel, out_channel in list(zip(down_filters, down_filters[1:])):
            self.down_blocks.append(TripleBlock(in_channel, out_channel))

        #UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for in_channel, out_channel in list(zip(up_filters, up_filters[1:])):
            self.up_blocks.append(TripleBlock(in_channel+out_channel, out_channel))

        # HELPERS
        self.pool = nn.MaxPool2d(2, stride=2)
        self.last_conv = nn.Conv2d(nfilters[0], 1, 3, padding=1)

        # FULLY CONNECTED LAYERS (conv layers that reduce to 1x1)
        self.fc = nn.Sequential(
            nn.Conv2d(nfilters[-1], nfilters[-1], 4),
            nn.BatchNorm2d(nfilters[-1]),
            nn.ReLU(True),
            nn.Conv2d(nfilters[-1], nfilters[-1], 1),
            nn.BatchNorm2d(nfilters[-1]),
            nn.ReLU(True),
            nn.Conv2d(nfilters[-1], bSize+LightVectorSize, 1),
        )

    def forward(self, x):
        #CONV_DOWN_BLOCKS
        x_skip = {}
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)

            if i != len(self.down_blocks)-1: #dont skip and pool last block
                x_skip[i] = x
                x = self.pool(x)
            
        y = x #store for later use
        # print([x.shape for x in x_skip.values()])

        z = None
        # Four decoders
        for _ in range(4):
            x = y   
            for i in range(len(self.up_blocks)):
                x = nn.Upsample(scale_factor=2)(x)#something upsample idk what
                # print(x.shape)
                # print(x_skip[len(self.down_blocks)-2-i].shape)
                x = torch.cat((x, x_skip[len(self.down_blocks)-2-i]), dim=1) #cat with x_skip[len(up_blocks)-i-1
                x = self.up_blocks[i](x)
            x = self.last_conv(x)
            z = x if z is None else torch.cat((x,z), dim=1)

        
        #Fully Connected Layers for estimation of lightvector and b
        p = self.fc(y)

        return z, p


