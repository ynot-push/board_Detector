import torch
import torch.nn as nn

def conv_block(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class BoardDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = conv_block(3, 16)      
        self.block2 = conv_block(16, 32)   
        self.block3 = conv_block(32, 64)    
        self.block4 = conv_block(64, 128)   
        self.block5 = conv_block(128, 256)   
        self.pool = nn.AdaptiveAvgPool2d(4)

        feat = 256 * 4 * 4
        self.board_head = nn.Linear(feat, 1)
        self.box_head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x) 
        x = torch.flatten(x, 1)
        board_logit = self.board_head(x)
        box = torch.sigmoid(self.box_head(x))
        return board_logit, box



if(__name__ == "__main__"):
    model = BoardDetectorCNN()
    logit, box = model(torch.randn(2, 3, 256, 256))
    print(logit.shape, box.shape) 
    