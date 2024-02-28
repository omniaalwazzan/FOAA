
import torch
import torch.nn as nn
import math as m
from torchvision import models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn.functional as F
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
class MultiHeadAttention_mul(nn.Module):
    def __init__(self, d_model=8, num_heads=4, dropout=0.1):
        super().__init__()

        self.d = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.mha_linear = nn.Linear(d_model, d_model)
    def scaled_dot_product_attention(self, Q, K, V):
        Q_K_matmul = torch.matmul(Q, K.transpose(-2, -1))
        scores = Q_K_matmul/m.sqrt(self.d)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, x):
        Q = [linear_Q(x) for linear_Q in self.linear_Qs]
        K = [linear_K(x) for linear_K in self.linear_Ks]
        V = [linear_V(x) for linear_V in self.linear_Vs]

        output_per_head = []
        attn_weights_per_head = []
        for Q_, K_, V_ in zip(Q, K, V):
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3) # we can return this for a visulaiztion purpose
        projection = self.dropout(self.mha_linear(output))

        return projection
    

class convNext(nn.Module):
    def __init__(self, n_classes=64):
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



class stand_atten_img(nn.Module):
    def __init__(self, model_image, d_model = 64 ,nb_classes=2):
        super(stand_atten_img, self).__init__()
        self.model_image =  model_image

        self.attention_img_mul =  MultiHeadAttention_mul(d_model=64, num_heads=1) #The number of heads can be adjusted, but it must be divisible by the model's dimension-->(64)
        self.fc = nn.Linear(128, 1000)
        self.dropout = nn.Dropout(p=0.1) #0.1
        self.layer_out = nn.Linear(1000, nb_classes)


    def forward(self, x1):
        x1 = self.model_image(x1)
        atten_img_mul = self.attention_img_mul(torch.unsqueeze(x1, 1))
        x1 = torch.unsqueeze(x1, 1)
        x_cat_channel = torch.cat((x1,atten_img_mul), dim=2)
        x = x_cat_channel.flatten(start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
    
#%%
from torchinfo import summary
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
img = convNext()
model = stand_atten_img(img)
model.to(device=DEVICE,dtype=torch.float)
print(summary(model,(2,3, 224, 224)))

