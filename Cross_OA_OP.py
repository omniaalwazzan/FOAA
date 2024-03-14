import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from Cross_Attention import cross_att
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
        self.layer_1 = Linear_Layer(3, 20) 
        self.layer_2 = Linear_Layer(20, 10)
        self.layer_3 = Linear_Layer(10, h_dim)
        self.dropout = nn.Dropout(p=0.01)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
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

class Cross_OA_OP(nn.Module):
    def __init__(self,atttention_OA,atttention_OP,model_image,model_EHR,nb_classes=2, h_dim=1):
        super(Cross_OA_OP, self).__init__()
        
        self.model_image =  model_image
        self.model_EHR = model_EHR
        self.atttention_OA = atttention_OA
        self.atttention_OP = atttention_OP

        self.att1 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for addition operation
        self.att2 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for addition operation
        
        self.att3 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for product operation
        self.att4 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for product operation
        
        self.fc1 = nn.Linear(64, 30) 
        self.ln =  nn.LayerNorm(30)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.01)
        self.layer_out = nn.Linear(30, nb_classes) 
    
    def forward(self, x1,x2):
        ### 1) recieve feature maps    
        x1 = self.model_image(x1)

        x2 = self.model_EHR(x2)
        x2 = torch.squeeze(x2, 1)
        ''' we need to have our featur map to be of size (bs,feature_dim,1) to work with moab,thus we unsqueeze'''
        x1 = torch.unsqueeze(x1, 2) 
        x2 = torch.unsqueeze(x2, 2)
        
        ### 2) Cross Attention Outer Addition
        I_prim_add = self.att1(x1,x2 )  
        EHR_prim_add = self.att2(x2,x1) 
    
        ### 3) Cross Attention Outer Product
        I_prim_prod = self.att3(x1,x2 ) 
        EHR_prim_prod = self.att4(x2,x1)
    
        ### 4) Aggregate enhanced features 
        x = torch.sum(torch.stack([I_prim_add,EHR_prim_add,I_prim_prod,EHR_prim_prod,x1,x2]), dim=0) # This operation will perform element-wise addition, maintaining the original size of the vector.
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
atttention_OA = 'OA'
atttention_OP = 'OP' 
ehr = EHR(EHR_out_dim)
img = convNext(CNN_out_dim)
model = Cross_OA_OP(atttention_OA,atttention_OP,img,ehr).to(device=DEVICE,dtype=torch.float)
print(summary(model,[(1,3, 64, 64),(1,3)]))


