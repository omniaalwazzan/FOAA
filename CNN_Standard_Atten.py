
import timm
import torch.nn as nn
import math as m
import torch
import torch.nn.functional as F
from torchvision import models
#from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
class MultiHeadAttention(nn.Module):
    # default values for the diminssion of the model is 8 and heads 4
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
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)

        projection = self.dropout(self.mha_linear(output))

        return projection#, attn_weights


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
    def __init__(self, cnn_img,h_dim,nb_classes=2):
        super(stand_atten_img,self).__init__()
        self.model_image =  cnn_img
        self.attention = MultiHeadAttention(d_model=1, num_heads=1)
        self.fc = nn.Linear(h_dim*1, 1000)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(1000, nb_classes)


    def forward(self, x):

        x = self.model_image(x)
        x = torch.unsqueeze(x, 2) # We use unsqueeze because standard attention mechanisms are designed to operate on 2D matrices
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


#%%
from torchinfo import summary
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
in_dim= 64
img = convNext()
model = stand_atten_img(img,in_dim)
model.to(device=DEVICE,dtype=torch.float)
print(summary(model,(2,3, 224, 224)))
