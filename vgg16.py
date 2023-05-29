import torch
import torch.nn as nn

torch.manual_seed(123)

class VGG16(nn.Module):
    def __init__(self, class_n):
        super().__init__()
               
        # 224 x 224
        #NOTE: padding = kernel_size // 2
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, (3,3), padding=1), 
                                    nn.GroupNorm(8,64),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, (3,3), padding=1), 
                                    nn.GroupNorm(8,64),
                                    nn.LeakyReLU(), 
                                    nn.MaxPool2d((2,2), stride=2))
        # 112 x 112
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, (3,3), padding=1), 
                                    nn.GroupNorm(16,128),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, (3,3), padding=1), 
                                    nn.GroupNorm(16,128),
                                    nn.LeakyReLU(), 
                                    nn.MaxPool2d((2,2), stride=2))
        # 56 x 56
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, (3,3), padding=1), 
                                    nn.GroupNorm(32,256),
                                    nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, (3,3), padding=1), 
                                    nn.GroupNorm(32,256),
                                    nn.LeakyReLU())
        self.layer7 = nn.Sequential(nn.Conv2d(256, 256, (3,3), padding=1), 
                                    nn.GroupNorm(32,256),
                                    nn.LeakyReLU(), 
                                    nn.MaxPool2d((2,2), stride=2))   
        # 28 x 28
        self.layer8 = nn.Sequential(nn.Conv2d(256, 512, (3,3), padding=1), 
                                    nn.GroupNorm(64,512),
                                    nn.LeakyReLU())
        self.layer9 = nn.Sequential(nn.Conv2d(512, 512, (3,3), padding=1), 
                                    nn.GroupNorm(64,512),
                                    nn.LeakyReLU())
        self.layer10 = nn.Sequential(nn.Conv2d(512, 512, (3,3), padding=1), 
                                     nn.GroupNorm(64,512),
                                     nn.LeakyReLU(), 
                                     nn.MaxPool2d((2,2), stride=2))    
        # 14 x 14
        self.layer11 = nn.Sequential(nn.Conv2d(512, 512, (3,3), padding=1), 
                                     nn.GroupNorm(64,512),
                                     nn.LeakyReLU())
        self.layer12 = nn.Sequential(nn.Conv2d(512, 512, (3,3), padding=1), 
                                     nn.GroupNorm(64,512),
                                     nn.LeakyReLU())
        self.layer13 = nn.Sequential(nn.Conv2d(512, 512, (3,3), padding=1), 
                                     nn.GroupNorm(64,512),
                                     nn.LeakyReLU(), 
                                     nn.MaxPool2d((2,2), stride=2)) 
        # 7 x 7
        self.layer14 = nn.Sequential(nn.Flatten(start_dim=1),
                                     nn.Linear(7*7*512, 4096),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5))
        self.layer15 = nn.Sequential(nn.Linear(4096, 4096),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5))
        self.layer16 = nn.Sequential(nn.Linear(4096, class_n))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        return out