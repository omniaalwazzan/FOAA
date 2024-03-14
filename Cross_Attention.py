import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

'''
Initially, we experimented with multihead attention in our code, but it's quite time-consuming and its imporovment was marginal. Therefore, I am opting to configure this code to accommodate both multihead and single head attention mechanisms.
We were only able to utilize 64 heads (feature dim) to align with the behavior of MOAB. This constraint may be one of the limitations of MOAB that we plan to tackle in the future. 
Additionally, while there are simpler methods for implementing attention, we favor explicit coding for its clarity and ease of comprehension.
''' 
def outer_add(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  + K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def outer_sub(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  - K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def outer_pro(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  * K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def mask_div(x1):
    mask = ((x1  > 0).float() - 1) * 9999  # for -inf
    result = (x1 + mask).softmax(dim=-1)
    return result   

def outer_div(Q, K):
    K = mask_div(K)
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  / K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

class cross_att(nn.Module):
    # default dim of the model is 8 and head is 4
    def __init__(self, attention_type, d_model=8, num_heads=4, dropout=0.01):
        super().__init__()
        self.d = d_model//num_heads
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.dropout = nn.Dropout(dropout)
        ##create a list of layers for K, and a list of layers for V for each head
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.mha_linear = nn.Linear(d_model, d_model)

    def attention(self, Q, K, V):
        if self.attention_type == 'OA': 
            scores = outer_add(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OP':
            scores = outer_pro(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OD':
            scores = outer_div(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OS':
            scores = outer_sub(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        else:
            print('attention type has not been selected')
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
# This example serves as a simple demonstration illustrating the application of our method in various fusion tasks. 

#For MHA
x1 = torch.randn(1,1,3) # heads should always match the model dim
x2 = torch.randn(1,1,3)
Arithmetic_operation_list = ['OA','OP','OD','OS']
attention_type = Arithmetic_operation_list[3]# This |
projection = cross_att(attention_type,d_model=3, num_heads=3)
ex1 = projection(x1,x2)

#%%

# For one head
x1 = torch.randn(1,3,1) 
x2 = torch.randn(1,3,1)
Arithmetic_operation_list = ['OA','OP','OD','OS']
attention_type = Arithmetic_operation_list[0] 
projection = cross_att(attention_type,d_model=1, num_heads=1)
ex1 = projection(x1,x2)
