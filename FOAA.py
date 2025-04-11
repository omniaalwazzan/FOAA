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

class FOAA(nn.Module):
    def __init__(self,atttention_OA,atttention_OP,atttention_OS,atttention_OD,model_image,model_EHR,nb_classes=2, h_dim=1):
        super(FOAA, self).__init__()
        
        self.model_image =  model_image
        self.model_EHR = model_EHR
        self.atttention_OA = atttention_OA
        self.atttention_OP = atttention_OP
        self.atttention_OS = atttention_OS
        self.atttention_OD = atttention_OD


        self.att1 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for addition operation
        self.att2 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for addition operation
        
        self.att3 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for product operation
        self.att4 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for product operation
        
        self.att5 = cross_att(atttention_OS,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for subtraction operation
        self.att6 = cross_att(atttention_OS,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for subtraction operation
        
        self.att7 = cross_att(atttention_OD,d_model=h_dim, num_heads=h_dim) # this to derive key from Img modality for subtraction operation
        self.att8 = cross_att(atttention_OD,d_model=h_dim, num_heads=h_dim) # this to derive key from EHR modality for subtraction operation
        
        
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
        I_prim_add = self.att1(x1,x2)  
        EHR_prim_add = self.att2(x2,x1) 
    
        ### 3) Cross Attention Outer Product
        I_prim_prod = self.att3(x1,x2) 
        EHR_prim_prod = self.att4(x2,x1)
        
        ### 4) Cross Attention Outer Subtraction
        I_prim_sub = self.att5(x1,x2) 
        EHR_prim_sub = self.att6(x2,x1)
        
        ### 5) Cross Attention Outer Division
        I_prim_div = self.att5(x1,x2) 
        EHR_prim_div = self.att6(x2,x1)
    
        ### 6) Aggregate FOAA enhanced features 
        x = torch.sum(torch.stack([I_prim_add,EHR_prim_add,I_prim_prod,EHR_prim_prod,I_prim_sub,EHR_prim_sub,I_prim_div,EHR_prim_div,x1,x2]), dim=0) # This operation will perform element-wise addition, maintaining the original size of the vector.
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
atttention_OS = 'OS'
atttention_OD = 'OD'
ehr = EHR(EHR_out_dim)
img = convNext(CNN_out_dim)
model = FOAA(atttention_OA,atttention_OP,atttention_OS,atttention_OD,img,ehr).to(device=DEVICE,dtype=torch.float)
print(summary(model,[(1,3, 64, 64),(1,3)]))

#%%
from fvcore.nn import FlopCountAnalysis
model.eval()
# Simulate inputs
x_img = torch.randn(1, 3, 64, 64).to(DEVICE)
x_omic = torch.randn(1, 3).to(DEVICE)

# Count FLOPs
with torch.no_grad():
    flops = FlopCountAnalysis(model, (x_img, x_omic))
    print(f"Total FLOPs: {flops.total():,}")
    print("FLOPs by module:")
    print(dict(flops.by_module()))
