import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from FOAA_SelfAttention import FOAA_SA


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

class CNN_FOAA_SA(nn.Module):
    def __init__(self,atttention_OA,atttention_OP,atttention_OS,atttention_OD,model_image,nb_classes=2, h_dim=1):
        super(CNN_FOAA_SA, self).__init__()
        
        self.model_image =  model_image
        self.atttention_OA = atttention_OA
        self.atttention_OP = atttention_OP
        self.atttention_OS = atttention_OS
        self.atttention_OD = atttention_OD

        self.att1 = FOAA_SA(atttention_OA,d_model=h_dim, num_heads=h_dim) 
        self.att2 = FOAA_SA(atttention_OP,d_model=h_dim, num_heads=h_dim)
        self.att3 = FOAA_SA(atttention_OS,d_model=h_dim, num_heads=h_dim)   
        self.att4 = FOAA_SA(atttention_OD,d_model=h_dim, num_heads=h_dim) 
                
        self.fc1 = nn.Linear(64, 30) 
        self.ln =  nn.LayerNorm(30)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.01)
        self.layer_out = nn.Linear(30, nb_classes) 
    
    def forward(self, x1):
        ### 1) recieve feature maps    
        x1 = self.model_image(x1)

        ''' we need to have our featur map to be of size (bs,feature_dim,1) to work with moab,thus we unsqueeze'''
        x1 = torch.unsqueeze(x1, 2) 
        
        ### 2) Self Attention Outer Addition
        I_prim_add = self.att1(x1)  
    
        ### 3) Self Attention Outer Product
        I_prim_prod = self.att2(x1) 
        
        ### 4) Self Attention Outer Subtraction
        I_prim_sub = self.att3(x1) 
        
        ### 5) Self Attention Outer Division
        I_prim_div = self.att4(x1) 
    
        ### 6) Aggregate CNN FOAA Self Attetnion enhanced features 
        x = torch.sum(torch.stack([I_prim_add,I_prim_prod,I_prim_sub,I_prim_div,x1]), dim=0) # This operation will perform element-wise addition, maintaining the original size of the vector.
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.ln(x)

        x = self.dropout(x)
        x = self.act(x)
        x = self.layer_out(x)

        return x    

#%%
CNN_out_dim = 64
atttention_OA = 'OA'
atttention_OP = 'OP' 
atttention_OS = 'OS'
atttention_OD = 'OD'
img = convNext(CNN_out_dim)
model = CNN_FOAA_SA(atttention_OA,atttention_OP,atttention_OS,atttention_OD,img).to(device=DEVICE,dtype=torch.float)
print(summary(model,(1,3, 64, 64)))


