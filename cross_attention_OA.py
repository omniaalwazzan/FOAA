import torch
import torch.nn as nn
import math as m

import torch.nn.functional as F
from torchvision import models
import numpy as np
import timm 
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn import init
from torchinfo import summary 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from Outer_addition_attention import *
#%%

class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)
    
class EHR(nn.Module):
    def __init__(self, h_dim):
        super(EHR, self).__init__()
        self.layer_1 = Linear_Layer(3, 20) #20
        self.layer_2 = Linear_Layer(20, 10) #10
        self.layer_3 = Linear_Layer(10, h_dim)
        self.dropout = nn.Dropout(p=0.01)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
        #x = self.dropout(x)
        return x

#%%

class convNext(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        convNext.avgpool = nn.AdaptiveAvgPool2d((1))
        convNext.classifier = nn.Sequential(nn.Flatten(1, -1),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(in_features=1024, out_features=n_classes))
        self.base_model = convNext

    def forward(self, x):
        x = self.base_model(x)
        return x
#%%

class FOAA_OA(nn.Module):
    def __init__(self,model_image,model_gens,nb_classes=2, h_dim=1):
        super(FOAA_OA, self).__init__()
        
        self.model_image =  model_image
        self.model_gens = model_gens
        self.att1 = cross_att_OA(d_model=h_dim, num_heads=h_dim)
        self.att5 = cross_att_OA(d_model=h_dim, num_heads=h_dim)
        self.fc1 = nn.Linear(64, 30) 
        self.ln =  nn.LayerNorm(30)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.01)
        self.layer_out = nn.Linear(30, nb_classes) 
    
    def forward(self, x1,x3):
        ### 1) recieve feature maps    
        x1 = self.model_image(x1)

        x3 = self.model_gens(x3)
        x3 = torch.squeeze(x3, 1)
        ''' we need to have our featur map to be of size (bs,feature_dim,1) to work with moab,thus we unsqueeze'''
        x1 = torch.unsqueeze(x1, 2) 
        x3 = torch.unsqueeze(x3, 2)
        
        ### 2) Cross Attention 
        I_prim_add = self.att1(x1,x3 )#.squeeze(2) # q from genes =x3, k & v from img =x1
        G_prim_add = self.att5(x3,x1)
    
        ### 3) Aggregate enhanced features 
        x = torch.sum(torch.stack([I_prim_add,G_prim_add,x1,x3]), dim=0) #this will do an ement wise addition, thus vector size didn't change
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.ln(x)

        x = self.dropout(x)
        x = self.act(x)
        x = self.layer_out(x)

        return x    

#%%
EHR_out_dim = 64
CNN_out_dim = 64
ehr = EHR(EHR_out_dim)
img = convNext(CNN_out_dim)
model = FOAA_OA(img,ehr).to(device=DEVICE,dtype=torch.float)
print(summary(model,[(1,3, 224, 224),(1,3)]))



