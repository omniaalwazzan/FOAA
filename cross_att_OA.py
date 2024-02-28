import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

'''
Initially, we experimented with multihead attention in our code, but it proved to be time-consuming.
We were only able to utilize 64 heads (feature dim) to align with the behavior of MOAB. This constraint may be one of the limitations of MOAB that we plan to tackle in the future. 
Therefore, I am opting to configure this code to accommodate both multihead and single head attention mechanisms.
Additionally, while there are simpler methods for implementing attention, we favor explicit coding for its clarity and ease of comprehension.
''' 
def outer_add(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  + K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul#torch.matmul(Q, K.transpose(-2, -1))

class cross_att_OA(nn.Module):
    # default dim of the model is 8 and head is 4
    def __init__(self, d_model=8, num_heads=4, dropout=0.01):
        super().__init__()
        self.d = d_model//num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        ##create a list of layers for K, and a list of layers for V for each head
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.mha_linear = nn.Linear(d_model, d_model)

    def attention(self, Q, K, V):
        scores = outer_add(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x,x2):

        Q = [linear_Q(x2) for linear_Q in self.linear_Qs] # Query is from modality x2 
        K = [linear_K(x) for linear_K in self.linear_Ks] # K & V are from x 
        V = [linear_V(x) for linear_V in self.linear_Vs]
        output_per_head = []
        attn_weights_per_head = []

        for Q_, K_, V_ in zip(Q, K, V):
            output, attn_weight = self.attention(Q_, K_, V_)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)


        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        projection = self.dropout(self.mha_linear(output))
        return projection

#%%

#For MHA
x1 = torch.randn(1,1,3) # heads should always match the model dim
x2 = torch.randn(1,1,3)
projection = cross_att_OA(d_model=3, num_heads=3)
ex1 = projection(x1,x2)

#%%

# For one head
x1 = torch.randn(1,3,1) 
x2 = torch.randn(1,3,1)
projection = cross_att_OA(d_model=1, num_heads=1)
ex1 = projection(x1,x2)
