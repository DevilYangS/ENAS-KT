from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
import copy



from numba import njit
from scipy.stats import rankdata

@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def Get_auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

# torch.nn.utils.weight_norm()

class Clone_Module(nn.Module):
    def __init__(self,module,N): # N is 2
        super(Clone_Module,self).__init__()
        self.module = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.N = N

        self.linear1= nn.Linear(128,128)
        self.linear2= nn.Linear(128,128)
        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p) # this initialization strategy is not suitable
                nn.init.kaiming_uniform_(p)
    def forward(self,inputs):
        encoder_embedding = self.module[0](inputs)
        decoder_embedding = self.module[1](inputs) if self.N >1 else self.module[0](inputs)
        return encoder_embedding,decoder_embedding
        # if isinstance(self.module[0], nn.Linear):
        #     return encoder_embedding,decoder_embedding
        # else:
        #     return self.linear1(encoder_embedding),self.linear2(decoder_embedding)



class Tags_set_embedding(nn.Module):
    def __init__(self,embedding_dim,tags_set_num,discrete =True):
        super(Tags_set_embedding, self).__init__()
        self.discrete = discrete
        if discrete :
            self.embedding = nn.Embedding(tags_set_num,embedding_dim)
        else:
            self.embedding = nn.Linear(tags_set_num,embedding_dim)
    def forward(self,x):
        device = self.embedding.weight.device
        if self.discrete:
            out = torch.sum(self.embedding(x[:,:,:7].to(device)),dim=-2)
            # out = 0
            # for i in range(7):
            #     out += self.embedding(x[:,:,i].to(device))
        else:
            out = self.embedding(x[:,:,7:].to(device).float())
        return out






class OutModule(nn.Module):
    def __init__(self,embedding_dim):
        super(OutModule, self).__init__()


        self.classfier_1 = nn.Linear(embedding_dim*3,embedding_dim)
        self.classfier_2 = nn.Linear(embedding_dim,1)
        self.activation1 = nn.GELU()
        self.activation2 = nn.Sigmoid()
        # self.LSTM = nn.LSTM(input_size=128,hidden_size=128,num_layers=2)

        pass
    def forward(self,encoder,decoder):

        x =torch.mul(encoder,self.activation2(decoder))

        x = torch.cat([x,encoder,decoder],dim=-1) # super 1
        # x = torch.cat([self.LSTM(decoder)[0],x],dim=-1) # super 1

        x = self.classfier_1(x)
        x = self.activation1(x)
        x = self.classfier_2(x)
        return x

#-----------------------------------------------------------------------------------------------

class combine_node(nn.Linear):
    def  __init__(self,num_candidate_feature,embed_dim,bias = True):
        super(combine_node, self).__init__(num_candidate_feature*embed_dim,embed_dim,bias = bias)
        self.embed_dim = embed_dim
        self.num_candidate_feature = num_candidate_feature

        # adding weight-sum
        self.Weight_Para1 = nn.Parameter(torch.randn(num_candidate_feature,100),requires_grad=True)
        # self.register_parameter('Weight_Para2', nn.Parameter(torch.randn(num_candidate_feature,embed_dim),requires_grad=True))

        self.activation = nn.Tanh()

        self.activationSet = nn.ModuleList([nn.ReLU(),nn.Sigmoid(),nn.Tanh(),nn.GELU()])





        uniform_ = None
        non_linear = 'linear'
        self._reset_parameters(bias, uniform_, non_linear)
    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.Weight_Para1) if uniform_ is None else uniform_(
            self.Weight_Para1, non_linear=non_linear)



    def forward(self,candidate_features,NAS):
        self.NAS = NAS

        x = [candidate_features[i] for i in self.NAS[:-1]]        #


        if self.NAS[-1]==0: # element-wise sum
            out =  sum(x)
        elif self.NAS[-1]==1: # concat-linear

            x = torch.cat(x,dim=-1)
            concat_weight = [self.weight[:,i*self.embed_dim:(i+1)*self.embed_dim] for i in self.NAS[:-1]]

            concat_weight = torch.cat(concat_weight,dim=-1)
            out = F.linear(x, concat_weight, self.bias)

        elif self.NAS[-1]==2: # weight-sum : encoding ==2
            xdim = 1
            x = [candidate_features[i].unsqueeze(1)  for i in self.NAS[:-1]]
            x = torch.cat(x,dim=xdim)

            sum_weight = F.softmax(self.Weight_Para1[self.NAS[:-1]],dim = 0)
            # sum_weight = self.Weight_Para1[self.NAS[:-1]==1]

            feature = torch.mul(x,sum_weight.unsqueeze(-1))
            out =  torch.sum(feature,dim=1)

        else :# dot : encoding ==3

            # return torch.mul(candidate_features[self.NAS[0]],candidate_features[self.NAS[1]]) #nn.Tanh()
            out =  torch.mul(candidate_features[self.NAS[0]],self.activation(candidate_features[self.NAS[1]])) #

        return self.activationSet[0](out)


class C_block1(nn.Linear): # Combine block
    def  __init__(self,num_candidate_feature,embed_dim,bias = True, Encoder_input =True):
        super(C_block1, self).__init__(num_candidate_feature*embed_dim,embed_dim,bias = bias)
        self.embed_dim = embed_dim
        self.num_candidate_feature = num_candidate_feature
        self.encoder_input = Encoder_input
        # adding weight-sum
        self.Weight_Para1 = nn.Parameter(torch.randn(num_candidate_feature,100),requires_grad=True)
        # self.register_parameter('Weight_Para2', nn.Parameter(torch.randn(num_candidate_feature,embed_dim),requires_grad=True))

        uniform_ = None
        non_linear = 'linear'
        self._reset_parameters(bias, uniform_, non_linear)
    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.Weight_Para1) if uniform_ is None else uniform_(
            self.Weight_Para1, non_linear=non_linear)

    def forward(self,candidate_features,NAS ):


        self.NAS = NAS

        if self.encoder_input:
            self.NAS[:2]=1 # exercise and concept part default into encoder
        else:
            self.NAS[2] = 1 # answer of response default into decoder


        x = [candidate_features[i] for i,idx in enumerate(self.NAS[:-1]) if idx==1]

        if self.NAS[-1]==0: # element-wise sum
            return sum(x)
        elif self.NAS[-1]==1: # concat-linear
            x = torch.cat(x,dim=-1)
            concat_weight = [self.weight[:,i*self.embed_dim:(i+1)*self.embed_dim] for i,idx in enumerate(self.NAS[:-1]) if idx]
            concat_weight = torch.cat(concat_weight,dim=-1)
            return F.linear(x, concat_weight, self.bias)
        else: # weight-sum : encoding ==2
            xdim = 1
            x = [candidate_features[i].unsqueeze(1) for i,idx in enumerate(self.NAS[:-1]) if idx==1]
            x = torch.cat(x,dim=xdim)

            sum_weight = F.softmax(self.Weight_Para1[self.NAS[:-1]==1],dim = 0)
            # sum_weight = self.Weight_Para1[self.NAS[:-1]==1]

            feature = torch.mul(x,sum_weight.unsqueeze(-1))
            return torch.sum(feature,dim=1)

class Combine_Block(nn.Module):
    def __init__(self,num_candidate_feature,embed_dim,num_of_node = 6,bias = True, Encoder_input =True):
        super(Combine_Block, self).__init__()

        self.encoder_input = Encoder_input
        self.num_of_node = num_of_node
        self.num_candidate_feature = num_candidate_feature

        self.Nodes = nn.ModuleList([])
        for i in range(num_of_node):
            self.Nodes.extend([combine_node(num_candidate_feature+i,embed_dim,bias = bias)])

        self.final_block = C_block1(num_candidate_feature = num_candidate_feature+num_of_node,embed_dim=embed_dim,bias = bias, Encoder_input =Encoder_input)

    def get_Sub_uniform_NAS_Node(self,num_candidate_feature):
        NAS = np.random.choice(num_candidate_feature,2,replace=False)

        randI = np.random.rand()
        if randI<0.1: #  【0， 0.2  0.6 1】
            insert = 0
        elif randI>=0.1 and randI<0.4:
            insert = 1
        elif randI>=0.4 and randI<0.7:
            insert = 2
        else:
            insert = 3
        NAS = np.hstack([NAS,insert])
        return NAS

    def get_final_block_uniform_NAS(self,num_candidate_feature):

        NAS = np.zeros([num_candidate_feature,])
        while NAS.sum()==0:
            NAS = np.random.randint(0,2,num_candidate_feature)
        randI = np.random.rand()

        # NAS = np.zeros([num_candidate_feature,])

        if randI<0.1: #  【0， 0.2  0.6 1】
            insert = 0
        elif randI>=0.1 and randI<=0.55:
            insert = 1
        else:
            insert = 2
        NAS = np.hstack([NAS,insert])


        return NAS


    def get_uniform_NAS(self):
        self.NAS = []
        for i in range(self.num_of_node):
            self.NAS.append(self.get_Sub_uniform_NAS_Node(self.num_candidate_feature+i))
        NAS2 = self.get_final_block_uniform_NAS(self.num_candidate_feature+self.num_of_node)
        # --------------------- deal with constraint------------------
         # last out put was used
        index = []
        for x in self.NAS:
            index.extend(x[:-1])
        index = np.unique(np.array(index))

        candidate_index = np.array(range(self.num_candidate_feature,self.num_candidate_feature+self.num_of_node))
        NAS2  =np.array(NAS2)
        index = np.setdiff1d(candidate_index,index)
        NAS2[index] = 1
        # --------------------- deal with constraint------------------
        self.NAS.append(NAS2)

    def forward(self,candidate_features,NAS=None):

        if NAS is not None:
            self.NAS = NAS
        else:
            self.get_uniform_NAS()
        # X = candidate_features[:3]


        for idx,module in enumerate(self.Nodes):
            output_feature = module(candidate_features,NAS = self.NAS[idx])
            candidate_features.append(output_feature)
            # X.append(output_feature)

        out = self.final_block(candidate_features,NAS = self.NAS[-1])
        return out








#-----------------------------------------------------------------------------------------------

class maskConV1d_V1(nn.Conv1d):
    def __init__(self,in_channels,out_channels,kernel_size,padding): # k [3,5,7,11]
        super(maskConV1d_V1, self).__init__(in_channels,out_channels,kernel_size=kernel_size,padding=padding,bias=False)
        self.register_buffer('mask', self.weight.data.clone())
        out_c,in_c, kernel= self.weight.size()
        assert kernel==kernel_size[0]
        self.mask.fill_(1)
        self.mask[:,:,kernel//2 +1 :] = 0
    def forward(self, x) :
        self.weight.data *= self.mask
        return super(maskConV1d_V1, self).forward(x)


#------------------------------------------------------- split -----------------------------------------------------------




class maskConV1d(nn.Conv1d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation=(1,)):
        super(maskConV1d, self).__init__(in_channels,out_channels,kernel_size=kernel_size,padding=padding,dilation=dilation,bias=False)
        self.padding_len = padding
    def forward(self, x) :
        x = super(maskConV1d, self).forward(x)
        return x[:,:,:-self.padding_len]
        # return x[:,:,:-self.padding_len].contiguous()


class MaskConV1d(nn.Module):
    def __init__(self,in_channels,out_channels,k,dropout = 0.0,dilation=1): # k [3,5,7,11]
        super(MaskConV1d, self).__init__()
        padding_size = 2*(dilation-1)+k-1
        self.conv1 = maskConV1d(in_channels,out_channels = out_channels,kernel_size=(k,),padding=padding_size,dilation=(dilation,))
        self.activation = nn.ReLU()
        # self.conv2 = maskConV1d(out_channels,out_channels = in_channels,kernel_size=(k,),padding=padding_size,dilation=(dilation,))
        self.conv2 = nn.Conv1d(out_channels,out_channels=out_channels,kernel_size=1)

        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.conv1(x.permute(0,2,1))
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x.permute(0,2,1)


class MaskGateLinearUnitConv(nn.Module):
    def __init__(self,in_channels,out_channels,k=3,dropout = 0.0,dilation=1): # k [3,5,7,11]
        super(MaskGateLinearUnitConv, self).__init__()
        padding_size = 2*(dilation-1)+k-1
        out_channels = 2*in_channels
        self.conv1 = maskConV1d(in_channels,out_channels = out_channels,kernel_size=(k,),padding=padding_size,dilation=(dilation,))
        self.glu = nn.GLU()
        self.dropout = nn.Dropout(dropout)

        # self.conv2 = maskConV1d(in_channels,out_channels = in_channels,kernel_size=(k,),padding=padding_size,dilation=(dilation,))
        self.conv2 = nn.Conv1d(in_channels,out_channels=in_channels,kernel_size=1)

    def forward(self,x):
        x = self.conv1(x.permute(0,2,1))
        x = self.glu(x.permute(0,2,1))
        x=  self.dropout(x)
        x = self.conv2(x.permute(0,2,1)).permute(0,2,1)
        return x









def get_mask(seq_len, encoding=None):
    if encoding == None:
        encoding = np.random.randint(1,np.ceil((seq_len-1)/5).astype(int)+1,1)
        ## add this to device
        # from [1] to [(seq_len/2) / 5] [1~10]
        # float type , performance is not normal, exceeding to 0.9 AUC
        # return torch.from_numpy(np.triu(np.ones([seq_len, seq_len]), k=1) + np.tril(np.ones([seq_len, seq_len]),k=-(1 + encoding * 5)).astype('bool')).cuda()
        # bool type, performance is normal
    return torch.from_numpy((np.triu(np.ones([seq_len, seq_len]), k=1) + np.tril(np.ones([seq_len, seq_len]),k=-(1 + encoding * 5))).astype('bool'))
def get_concat_mask(seq_len,out,encoder_out,encoding = None):
    return torch.cat([get_mask(seq_len,encoding) for _ in range(int(encoder_out.shape[0]/out.shape[0]))],dim=1)

class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, n_dims):
        super(PositionEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, device):
        seq = torch.arange(self.seq_len).unsqueeze(0).to(device)
        p = self.position_embed(seq)
        return p

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[offset:x.size(0)+offset, :]
        return self.dropout(x)


class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout=0.1):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn)  # [batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model)  # [batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)  # here dropout is normal dropout
        x = self.linear_2(x)
        return x

class FFN1(nn.Module):
    def __init__(self, d_ffn, d_model, dropout=0.1):
        super(FFN1, self).__init__()
        self.linear_1 = nn.Linear(2*d_model, d_ffn)  # [batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model)  # [batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)  # here dropout is normal dropout
        x = self.linear_2(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x


class SelfAttention(nn.Module):
    def __init__(self,dim_model,heads = 8, dropout_rate = 0.0,max_length = 100):
        super(SelfAttention, self).__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads, dropout=dropout_rate)
        self.max_length = max_length

    def forward(self,x, encoder_out = None, NAS = 20):
        x = x.permute(1,0,2)
        if encoder_out is None:
            encoder_out = x
        else:
            encoder_out = encoder_out.permute(1,0,2)

        out, attn_wt = self.MHA(x, encoder_out, encoder_out, attn_mask=get_concat_mask(seq_len=self.max_length,out=x,encoder_out=encoder_out,encoding=NAS).to(x.device))
        return out.permute(1,0,2)




class GLU(nn.Module):
    def __init__(self,embed_dim):
        super(GLU, self).__init__()
        self.norm_glu = nn.LayerNorm(embed_dim)
        self.glu_ff1 = nn.Linear(embed_dim,embed_dim)
        self.activation = nn.Sigmoid()
        self.glu_ff2 = nn.Linear(embed_dim,embed_dim)
    def forward(self,x):
        x = self.norm_glu(x)
        values = self.glu_ff1(x)
        gates = self.activation(self.glu_ff2(x))
        hidden_state = values*gates
        return hidden_state




class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)





class BCESmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCESmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, predictions, bi_labels=None):
        if bi_labels is None:
            return torch.tensor([0.0])
        smooth_labels = bi_labels * (1-self.smoothing) + (self.smoothing/bi_labels.size(-1))
        loss = self.bce(predictions, smooth_labels)
        return loss

class BCESmoothLossOnevsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCESmoothLossOnevsAll, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        with torch.no_grad():
            weight = predictions.new_ones(predictions.size()) * self.smoothing / (predictions.size(-1) - 1.)
            weight.scatter_(-1, labels.unsqueeze(-1), (1. - self.smoothing))
        loss = self.bce(predictions, weight)
        return loss





class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()
    def forward(self, predictions, bi_labels=None):
        if bi_labels is None:
            return torch.tensor([0.0])
        smooth_labels = bi_labels * (1.0 - self.smoothing) + 0.5 * self.smoothing
        # smooth_labels = bi_labels * (1-self.smoothing) + (self.smoothing/bi_labels.size(-1)) # bi_labels.size(-1) is 2

        loss = self.bce(predictions, smooth_labels)
        return loss



