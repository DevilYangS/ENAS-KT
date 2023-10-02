from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from Operations import *
import re

num_of_Operatopn = 3
num_of_LocalOperatopn = 5



class Transformer_super_V3(nn.Module):
    def __init__(self,config):
        super(Transformer_super_V3, self).__init__()
        self.config = config
        self.device = config.device
        self.dataset = config.dataset
        self.evalmodel = config.evalmodel

        self.num_en = num_en = 4
        self.num_de = num_de = 4
        self.DEC_HEADS = self.ENC_HEADS = DEC_HEADS = ENC_HEADS = 8

        EMBED_DIMS = config.embed_size
        self.EMBED_DIMS = EMBED_DIMS

        MAX_SEQ  = config.max_length
        #-----------------------------------------------------------------------------------
        dropout = config.dropout
        self.encoder = Transformer_Encoder(EMBED_DIMS, ENC_HEADS ,MAX_SEQ , num_en,dropout)
        self.decoder = Transformer_Decoder(EMBED_DIMS, DEC_HEADS ,MAX_SEQ , num_de,num_en,dropout)


        self.out = nn.Linear(in_features=EMBED_DIMS, out_features=1) # for Baseline: SAINT, etc.




        #-----------------------------------Loss ------------------------------------------------
        self.loss_BCE = nn.BCELoss()
        self.loss_BCE_1 = nn.BCELoss(reduction='none')
        self.loss_BCE_2 = BCESmoothLoss(smoothing=0.1)

        #-----------------------------------Position Embedding and other Embeddings---------------------------------
        self.encoder_position_embed = PositionEmbedding(MAX_SEQ, EMBED_DIMS)# posi_embed exists in Encoder_block or Decoder_block
        self.decoder_position_embed = PositionEmbedding(MAX_SEQ, EMBED_DIMS)# posi_embed exists in Encoder_block or Decoder_block


        # problem related
        self.exercise_embed = Clone_Module(nn.Embedding(config.data_info['problem_num'][-1] + 1,EMBED_DIMS),2)
        self.category_embed = Clone_Module( nn.Embedding(config.data_info['skill_num'][-1] + 1,EMBED_DIMS),2)
        self.tags_embed = Clone_Module( nn.Embedding(config.data_info['tags_num'][-1] + 1, EMBED_DIMS),2)

        self.tags_set_embed_1 = Clone_Module(Tags_set_embedding(tags_set_num=config.data_info['tags_set_num'][-1] + 1,embedding_dim=EMBED_DIMS,discrete=True),2)
        # self.tags_set_embed_2 = Clone_Module(Tags_set_embedding(tags_set_num=config.data_info['tags_set_num'][-1] + 1,embedding_dim=EMBED_DIMS,discrete=False),2)
        # label/correct embed
        self.response_embed = Clone_Module(nn.Embedding(3, EMBED_DIMS),2)
        # time
        self.lag_time_embed = Clone_Module(nn.Linear(1, EMBED_DIMS, bias=False),2)
        self.elapsed_time_embed = Clone_Module(nn.Linear(1, EMBED_DIMS, bias=False),2)

        self.decoder_etime_embed = Clone_Module(nn.Embedding(300 + 3, EMBED_DIMS),2)  # Elapsed time Embedding
        self.decoder_ltime_embed_s = Clone_Module(nn.Embedding(300 + 3, EMBED_DIMS),2)  # Lag time Embedding 1
        self.decoder_ltime_embed_m = Clone_Module(nn.Embedding(1440 + 3, EMBED_DIMS),2)  # Lag time Embedding 2
        self.decoder_ltime_embed_h = Clone_Module(nn.Embedding(365 + 3, EMBED_DIMS),2)  # Lag time Embedding 3

        if self.dataset=='Ednet':
            bundle_num = config.data_info['bundle_num'][-1] + 1
            self.bundle_embed = Clone_Module(nn.Embedding(bundle_num, EMBED_DIMS),2)
        elif self.dataset=='Riid':
            self.encoder_p_explan_embed = Clone_Module(nn.Embedding(2 + 2, EMBED_DIMS),2)


        # self.encoder_in = C_block(num_candidate_feature = 12,embed_dim=EMBED_DIMS,bias = True, Encoder_input =True)
        # self.decoder_in = C_block(num_candidate_feature = 12,embed_dim=EMBED_DIMS,bias = True, Encoder_input =False)

        self.encoder_in = Hierarchical_CBlock(num_candidate_feature = 12,embed_dim=EMBED_DIMS, Encoder_input =True)
        self.decoder_in = Hierarchical_CBlock(num_candidate_feature = 12,embed_dim=EMBED_DIMS, Encoder_input =False)

        self.init_weights()

    def init_weights(self):

        for n,p in self.named_parameters():
            if re.match(r'.*bias$|.*bn\.weight$|.*norm.*\.weight',n):
                continue
            gain = 1.
            if re.match(r'.*decoder.*',n):
                gain = (9*self.num_de)**(-1./4.)
                if re.match(f'.*in_proj_weight$',n): gain *= (2**0.5)
            elif re.match(r'.*encoder.*',n):
                gain = 0.67*(self.num_en**(-1./4.))
                if re.match(f'.*in_proj_weight$',n): gain *= (2**0.5)
            if re.match(r'.*embed.*', n):
                trunc_normal_(p.data,std=(4.5*(self.num_en+self.num_de))**(-1./4.)*self.EMBED_DIMS**(-0.5))
            else:
                nn.init.xavier_normal_(p,gain=gain)

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p) # this initialization strategy is not suitable
                # nn.init.kaiming_uniform_(p)
                # nn.init.xavier_normal(p)
                pass


    def loss(self,outdict):



        predictions = outdict['predictions']
        labels = outdict['labels']
        loss = self.loss_BCE(predictions, labels.float())

        return loss

    def forward(self,input_dict,NAScoding=None):

        if self.evalmodel == 'weight-sharing' and NAScoding is None: # uniform sampled

            NAScoding = [None,None,None,None,None]

        elif NAScoding=='maximal':
            NAScoding = ['maximal','maximal','maximal','maximal','maximal']
            pass
        elif NAScoding == 'minimal':
            NAScoding = ['minimal','minimal','minimal','minimal','minimal']


        elif NAScoding is None and self.evalmodel != 'weight-sharing' : # get NAScoding from config
            #---------------C_block---------------
            NAScoding_temp = self.config.NAS
            NAScoding = [np.array(x) for x in NAScoding_temp]

            # [encoder_input, decoder_input, encoder_coding, decoder_coding, decoder_cross-attention]

        #--------------------------------------------- handle inputdict ---------------------------------------------
        #--------------get data from input_dict--------------
        EL_TIME = input_dict['elapsed_time_seq_tensor'].numpy()
        LAG_TIME = input_dict['time_lag_tensor'].numpy()

        problem_seqs_tensor = input_dict['problem_seq_tensor'].to(self.device)
        skill_seqs_tensor = input_dict['skill_seq_tensor'].to(self.device)
        correct_seqs_tensor = input_dict['correct_seq_tensor'].to(self.device)
        tags_seqs_tensor = input_dict['tags_seq_tensor'].to(self.device)

        elapsed_time = (input_dict['elapsed_time_seq_tensor']*1e-5).to(self.device) if  self.dataset =='Ednet' or  self.dataset=='Riid' \
            else input_dict['timestamp_seq_tensor'].to(self.device)
        lag_time = (input_dict['time_lag_tensor']*1e-5).to(self.device)

        # --------------deal with label with padding first bit with '2' and  Move back one bit --------------
        mask_labels = correct_seqs_tensor * (correct_seqs_tensor < 2).long()
        input_label = mask_labels.clone()
        input_label[:, 0] = 2
        input_label[:, 1:] = mask_labels[:, :-1]
        mask_labels = input_label.long()




        #------------------------------------------------------------------------------------------------------
        position_embedding_encoder = self.encoder_position_embed(self.device)
        position_embedding_decoder = self.decoder_position_embed(self.device)

        problem_embedding = self.exercise_embed(problem_seqs_tensor)
        skill_embedding = self.category_embed(skill_seqs_tensor)
        response_embedding = self.response_embed(mask_labels)

        lag_time_embedding = self.lag_time_embed(lag_time.unsqueeze(-1).float())
        elapsed_time_embedding = self.elapsed_time_embed(elapsed_time.unsqueeze(-1).float())
        tags_embedding = self.tags_embed(tags_seqs_tensor)
        # tags_embedding = self.tags_linear(tags_seqs_tensor.unsqueeze(-1).float())

        etime_seq = self.decoder_etime_embed(( torch.Tensor(np.clip(EL_TIME//1000,0,300)).to(self.device) ).long())
        ltime_s_seq = self.decoder_ltime_embed_s( ( torch.Tensor(np.clip(LAG_TIME//1000,0,300)).to(self.device) ).long() )
        ltime_m_seq = self.decoder_ltime_embed_m(( torch.Tensor(np.clip(LAG_TIME//(1000*60),0,1440)).to(self.device) ).long())
        ltime_d_seq = self.decoder_ltime_embed_h( ( torch.Tensor(np.clip(LAG_TIME//(1000*60*1440),0,365)).to(self.device) ).long() )


        #-------------------------------------------------------------------------------------------------------

        # x0= [problem_embedding,skill_embedding,response_embedding,lag_time_embedding,elapsed_time_embedding,tags_embedding]
        x0= [problem_embedding[0],skill_embedding[0],response_embedding[0],lag_time_embedding[0],elapsed_time_embedding[0],tags_embedding[0]]
        x1= [problem_embedding[1],skill_embedding[1],response_embedding[1],lag_time_embedding[1],elapsed_time_embedding[1],tags_embedding[1]]
        if self.dataset=='Ednet':
            bundle_seqs_tensor = input_dict['bundle_seq_tensor'].to(self.device)
            bundle_embedding = self.bundle_embed(bundle_seqs_tensor)
            # x0.append(bundle_embedding)
            x0.append(bundle_embedding[0])
            x1.append(bundle_embedding[1])
        elif self.dataset=='Riid':
            p_explan_seq_tensor = input_dict['explanation_seq_tensor'].to(self.device)
            p_explan_embedding = self.encoder_p_explan_embed(p_explan_seq_tensor)
            # x0.append(p_explan_embedding)
            x0.append(p_explan_embedding[0])
            x1.append(p_explan_embedding[1])
        # x0.extend([etime_seq,ltime_s_seq,ltime_m_seq,ltime_d_seq])
        x0.extend([etime_seq[0],ltime_s_seq[0],ltime_m_seq[0],ltime_d_seq[0]])
        x1.extend([etime_seq[1],ltime_s_seq[1],ltime_m_seq[1],ltime_d_seq[1]])


        tags_set_embedding_1  = self.tags_set_embed_1(input_dict['tags_set_seq_tensor'])
        # tags_set_embedding_2  = self.tags_set_embed_2(input_dict['tags_set_seq_tensor'])
        x0.extend([tags_set_embedding_1[0]])
        x1.extend([tags_set_embedding_1[1]])


        #--------------------------------------------- handle inputdict ---------------------------------------------
        encoder_in = self.encoder_in(x0, NAS = NAScoding[0])+position_embedding_encoder
        decoder_in = self.decoder_in(x1, NAS = NAScoding[1])+position_embedding_decoder
        # ----------------------------------encoder-decoder----------------------
        encoder_out = self.encoder(encoder_in,NAS_coding=NAScoding[2])
        decoder_out  =self.decoder(decoder_in,encoder_out,NAS_coding = NAScoding[3],NAS_link = NAScoding[4])
        # ----------------------------------output predictions----------------------
        out = self.out(decoder_out)
        # out = self.out(encoder_out[-1],decoder_out)
        predictions = out.squeeze(-1).sigmoid()
        # ----------------------------------deal loss and label----------------------
        # TODO  first bit for loss but not for AUC/ACC
        loss_mask = (correct_seqs_tensor < 2)
        # predictions_loss = torch.masked_select(predictions, loss_mask)
        predictions_loss =predictions.clone()
        # labels_loss = torch.masked_select(correct_seqs_tensor, loss_mask)
        labels_loss = correct_seqs_tensor.clone()
        predictions = torch.masked_select(predictions[:,1:], loss_mask[:,1:])
        labels = torch.masked_select(correct_seqs_tensor[:,1:], loss_mask[:,1:])
        # out_dict = {'predictions': predictions, 'labels': labels}
        out_dict = {'predictions': predictions, 'labels': labels,'predictions_loss':predictions_loss,'labels_loss':labels_loss,'mask':loss_mask}
        return out_dict




class Clone_Module(nn.Module):
    def __init__(self,module,N): # N is 2
        super(Clone_Module,self).__init__()
        if isinstance(module,nn.Linear):
            self.module = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        elif isinstance(module,nn.Embedding):
            num_fea, dim =  module.num_embeddings,module.embedding_dim
            if dim<10:
                embedding_dim = round(dim/2)+1
            elif dim <100:
                embedding_dim = round(dim/10)+1
            elif dim <400:
                embedding_dim = round(dim/15)+1
            elif dim<3000:
                embedding_dim = round(dim/50)+1
                embedding_dim = round(dim/50)+1
            elif dim<10000:
                embedding_dim = round(dim/200)+1
            elif dim>=10000:
                embedding_dim = 60

            assert embedding_dim<dim
            mldule = nn.Embedding(num_fea,embedding_dim)
            self.module = nn.ModuleList([copy.deepcopy(mldule) for i in range(N)])
            self.linear1= nn.Linear(embedding_dim,dim)
            self.linear2= nn.Linear(embedding_dim,dim)
        else:
            self.module = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            self.linear1= nn.Linear(128,128)
            self.linear2= nn.Linear(128,128)

            # self.linear1= nn.Linear(256,256)
            # self.linear2= nn.Linear(256,256)

        self.N = N
        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p) # this initialization strategy is not suitable
                nn.init.kaiming_uniform_(p)
    def forward(self,inputs):
        encoder_embedding = self.module[0](inputs)
        decoder_embedding = self.module[1](inputs) if self.N >1 else self.module[0](inputs)
        # return encoder_embedding,decoder_embedding
        if isinstance(self.module[0], nn.Linear):
            return encoder_embedding,decoder_embedding
        else:
            return self.linear1(encoder_embedding),self.linear2(decoder_embedding)


class Hierarchical_CBlock(nn.Module):
    def __init__(self,num_candidate_feature,embed_dim, Encoder_input =True):
        super(Hierarchical_CBlock, self).__init__()
        self.num_candidate_feature = num_candidate_feature

        self.Bimodal_list = nn.ModuleList([])
        number_of_biFeature = round(num_candidate_feature*(num_candidate_feature-1)/2)
        self.number_of_biFeature = number_of_biFeature
        for i in range(number_of_biFeature):
            self.Bimodal_list.extend([Bimodal(embed_dim)])


        self.final_modal = FinalModal(number_of_biFeature, embed_dim)
        self.Zero = ZeroNormal()
        self.encoder_input = Encoder_input
    def get_uniform_NAS(self):
        self.NAS = np.zeros([self.num_candidate_feature,])
        while self.NAS.sum()==0:
            self.NAS = np.random.randint(0,2,self.num_candidate_feature)
    def forward(self,candidate_features,NAS = None):

        if isinstance(NAS,str):
                if NAS == 'maximal':
                    NAS = np.ones([self.num_candidate_feature,])
                elif NAS =='minimal':
                    NAS = np.zeros([self.num_candidate_feature,])
        elif NAS is None:
            self.get_uniform_NAS()
            NAS = self.NAS


        # NAS = NAS[:-1]

        if self.encoder_input:
            NAS[:2]=1 # exercise and concept part default into encoder
            # NAS[0]=1 # exercise and concept part default into encoder
        else:
            NAS[2] = 1 # answer of response default into decoder

        ItemSet = []
        IndexSet= []
        for idx,item in enumerate(NAS):
            if idx<len(NAS):
                for jdx,itemj in enumerate(NAS[idx+1:]):
                    if item==1 and itemj==1:
                        ItemSet.extend([1])
                        IndexSet.append([idx,idx+jdx+1])
                    else:
                        ItemSet.extend([0])
                        IndexSet.append([0,0])

        x = []
        for i, (module,index) in enumerate(zip(ItemSet,IndexSet)):
            if self.training:

                if module and np.random.rand()<0.9:
                # if module:
                    x.append(self.Bimodal_list[i](candidate_features[index[0]], candidate_features[index[1]]))
                else:
                    x.append(self.Zero(candidate_features[index[0]]))

            else:
                if module:
                    x.append(self.Bimodal_list[i](candidate_features[index[0]], candidate_features[index[1]]))
                else:
                    x.append(self.Zero(candidate_features[index[0]]))

        x = self.final_modal(x)


        return x


class Bimodal(nn.Module):
    def __init__(self,embed_dim):
        super(Bimodal, self).__init__()
        self.activation = nn.Tanh()
        self.linear_1 = nn.Linear(2*embed_dim,embed_dim,bias=True)
        # self.linear_1 = nn.Linear(2*embed_dim,32,bias=True)
    def forward(self,x1,y1):
        x = torch.cat([x1,y1],dim=-1)
        x = self.linear_1(x)
        return self.activation(x)

class FinalModal(nn.Linear):
    def __init__(self,num_candidate_feature,embed_dim):
        super(FinalModal, self).__init__(num_candidate_feature*embed_dim,embed_dim,bias=True)
        # super(FinalModal, self).__init__(num_candidate_feature*32,embed_dim,bias=True)
        self.embed_dim = embed_dim
        self.num_candidate_feature = num_candidate_feature

    def forward(self,x):

        x = torch.cat(x,dim=-1)

        return super(FinalModal, self).forward(x)

class C_block(nn.Linear): # Combine block
    def  __init__(self,num_candidate_feature,embed_dim,bias = True, Encoder_input =True):
        super(C_block, self).__init__(num_candidate_feature*embed_dim,embed_dim,bias = bias)
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

    def get_uniform_sampled_NAS(self):
        self.NAS = np.zeros([self.num_candidate_feature,])
        while self.NAS.sum()==0:
            self.NAS = np.random.randint(0,2,self.num_candidate_feature)


        randI = np.random.rand()
        if randI<0.1: #  【0， 0.2  0.6 1】
            insert = 0
        elif randI>=0.1 and randI<=0.55:
            insert = 1
        else:
            insert = 2
        self.NAS = np.hstack([self.NAS,insert])

    def forward(self,candidate_features,NAS = None):

        if NAS is not None:
            self.NAS = NAS
        else:
            self.get_uniform_sampled_NAS()

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

class Transformer_Encoder(nn.Module):
    def __init__(self, EMBED_DIMS, ENC_HEADS, MAX_SEQ, num_en,dropout_rate = 0.0):
        super().__init__()
        self.num_en = num_en
        self.seq_len = MAX_SEQ
        self.layers = nn.ModuleList([])
        self.layer_norm = nn.ModuleList([])  #  Layer Normalization of encoder_out used for decoder
        for i in range(num_en):
            self.layers.extend([
                Encoder_Block(EMBED_DIMS, ENC_HEADS, MAX_SEQ, dropout_FFN=dropout_rate, dropout_MHA=dropout_rate)])
            self.layer_norm.extend([nn.LayerNorm(EMBED_DIMS)])
            # self.layer_norm.extend([Identity()])
    def forward(self, x,NAS_coding = None):
        ## TODO  sampled mask for supermodel training
        if isinstance(NAS_coding,str):
            if NAS_coding=='maximal':
                NAS_coding = np.array([2,0,num_of_LocalOperatopn-1,2,0,num_of_LocalOperatopn-1,2,0,num_of_LocalOperatopn-1,
                                       2,0,num_of_LocalOperatopn-1])
            elif NAS_coding == 'minimal':
                NAS_coding = np.array([0,2,0, 0,2,0, 0,2,0, 0,2,0])
        elif NAS_coding is None:
            NAS_coding =np.random.randint(0,num_of_Operatopn,[self.num_en,2])
            NAS_coding1 =np.random.randint(0,num_of_LocalOperatopn,[self.num_en,1])
            NAS_coding = np.hstack([NAS_coding,NAS_coding1]).reshape([-1,])

        out = []
        for idx, layer in enumerate(self.layers):
            x = layer(x,NAS_coding = NAS_coding[idx*3:(idx+1)*3])
            append_x = self.layer_norm[idx](x) # only used for append_out, since each Block has layerNorm at its head but no layerNorm at its end
            out.append(append_x)
        return out

class Transformer_Decoder(nn.Module):
    def __init__(self, EMBED_DIMS, ENC_HEADS, MAX_SEQ, num_de, num_en,dropout_rate = 0.0):
        super().__init__()
        self.num_de = num_de
        self.num_en = num_en
        self.seq_len = MAX_SEQ
        self.layers = nn.ModuleList([])
        self.layer_norm_dec_end = nn.LayerNorm(EMBED_DIMS)
        # self.layer_norm_dec_end = Identity()
        for i in range(num_de):
            self.layers.extend([
                Decoder_Block(EMBED_DIMS, ENC_HEADS, MAX_SEQ, dropout_FFN=dropout_rate, dropout_MHA=dropout_rate)])

    def get_uniform_sampled_NAS(self):
        self.NAS = np.random.randint(0, 2, [self.num_de, self.num_en])

        if self.NAS[:, -1].sum() == 0:
            idx = np.random.choice(self.num_de, 1)
            self.NAS[idx, -1] = 1
        for idx in range(self.num_de):
            if self.NAS[idx].sum() == 0:
                random_idx = np.random.choice(self.num_en, 1)
                self.NAS[idx, random_idx] = 1
    def extract_from_encoder_out(self, idx, encoder_out):
        coding = self.NAS[idx]
        encoder_out_ = []
        for i, coding_i in enumerate(coding):
            if coding_i == 1:
                encoder_out_.append(encoder_out[i])
        encoder_out_ = torch.cat(encoder_out_, dim=1)
        return encoder_out_

    def forward(self, x, encoder_out, NAS_coding=None, NAS_link=None ):
        ##  to be consistent with original version
        ##   sampled mask for supermodel training
        if isinstance(NAS_coding,str):
            if NAS_coding=='maximal':
                NAS_coding = np.array([2,0,num_of_LocalOperatopn-1,2,0,num_of_LocalOperatopn-1,2,0,num_of_LocalOperatopn-1,
                                       2,0,num_of_LocalOperatopn-1])
            elif NAS_coding == 'minimal':
                NAS_coding = np.array([0,2,0, 0,2,0, 0,2,0, 0,2,0])
        elif NAS_coding is None:
            NAS_coding =np.random.randint(0,num_of_Operatopn,[self.num_de,2])
            NAS_coding1 =np.random.randint(0,num_of_LocalOperatopn,[self.num_de,1])
            NAS_coding = np.hstack([NAS_coding,NAS_coding1]).reshape([-1,])


        if isinstance(NAS_link,str):
            if NAS_link =='maximal':
                self.NAS = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
            elif NAS_link =='minimal':
                self.NAS = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]])
        elif  NAS_link is not None:
            self.NAS = NAS_link
        else:
            self.get_uniform_sampled_NAS()

        decoder_out = x
        for idx, layer in enumerate(self.layers):
            encoder_out_ = self.extract_from_encoder_out(idx, encoder_out)
            decoder_out = layer(decoder_out, encoder_out_,NAS_coding = NAS_coding[idx*3:(idx+1)*3])

        return self.layer_norm_dec_end(decoder_out)

def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"

    return x.normal_().fmod_(2).mul_(std).add_(mean)


class MixedOp(nn.Module):
    def __init__(self,args):
        super(MixedOp, self).__init__()
        self.OperationSet = nn.ModuleList([])
        self.OperationSet.extend([FFN(d_ffn=args['d_ffn'],d_model=args['dim_model'],dropout=args['dropout_FFN'])])
        self.OperationSet.extend([Zero() ])
        self.OperationSet.extend([SelfAttention(dim_model=args['dim_model'], heads=args['heads_en'], dropout_rate= args['dropout_MHA'],max_length=args['seq_len'])])
        # self.OperationSet.extend([SelfAttention(dim_model=args['dim_model'], heads=2, dropout_rate= args['dropout_MHA'],max_length=args['seq_len'])])
        # self.OperationSet.extend([SelfAttention(dim_model=args['dim_model'], heads=3, dropout_rate= args['dropout_MHA'],max_length=args['seq_len'])])
        # self.OperationSet.extend([SelfAttention(dim_model=args['dim_model'], heads=4, dropout_rate= args['dropout_MHA'],max_length=args['seq_len'])])
        # self.OperationSet.extend([SelfAttention(dim_model=args['dim_model'], heads=6, dropout_rate= args['dropout_MHA'],max_length=args['seq_len'])])
        # self.OperationSet.extend([FFN(d_ffn=2*args['d_ffn'],d_model=args['dim_model'],dropout=args['dropout_FFN'])])

        self.num_of_operation = len(self.OperationSet)
    def forward(self,x,NAS_coding=None):
        if NAS_coding is None:
            NAS_coding = np.random.choice(self.num_of_operation)

        x = self.OperationSet[NAS_coding](x)
        return x

class MixedOpLocal(nn.Module):
    def __init__(self,args):
        super(MixedOpLocal, self).__init__()
        self.OperationSet = nn.ModuleList([])

        self.OperationSet.extend([Zero() ])
        self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=3,dropout=args['dropout_FFN']) ])
        # self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=3,dropout=args['dropout_FFN'],dilation=2) ])
        # self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=3,dropout=args['dropout_FFN'],dilation=3) ])
        self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=5,dropout=args['dropout_FFN']) ])
        self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=7,dropout=args['dropout_FFN']) ])
        self.OperationSet.extend([MaskConV1d(in_channels=args['dim_model'],out_channels=args['dim_model'],k=11,dropout=args['dropout_FFN'])])


        self.num_of_operation = len(self.OperationSet)
    def forward(self,x,NAS_coding=None):
        if NAS_coding is None:
            NAS_coding = np.random.choice(self.num_of_operation)
        x = self.OperationSet[NAS_coding](x)
        return x

class ZeroNormal(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        return torch.zeros_like(x).to(x.device)
        # b,l,d =x.shape
        # return torch.zeros([b,l,32]).to(x.device)


class Zero(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        return torch.zeros_like(x).to(x.device)





class Encoder_Block(nn.Module):
    def __init__(self, dim_model, heads_en, seq_len, d_ffn=None, dropout_MHA=0.0, dropout_FFN=0.0, drop_path=0.0):
        super(Encoder_Block, self).__init__()
        self.seq_len = seq_len
        if d_ffn is None:
            d_ffn = 4 * dim_model
            d_ffn = dim_model

        args = {'d_ffn': d_ffn,
                'dim_model':dim_model,
                'heads_en':heads_en,
                'seq_len':seq_len,
                'dropout_MHA':dropout_MHA,
                'dropout_FFN':dropout_FFN,
                }

        self.layer_norm1 = nn.LayerNorm(dim_model)
        # self.layer_norm1 = Identity()
        self.layer_norm2 = nn.LayerNorm(dim_model)
        # self.layer_norm2 = Identity()




        # self.atten = SelfAttention(dim_model,heads=heads_en,dropout_rate= dropout_MHA,max_length=seq_len)
        # self.atten = MixedAttenConvOp(args)

        # self.ffn_en = FFN(d_ffn, dim_model, dropout=dropout_FFN)



        self.atten = MixedOp(args)
        self.ffn_en = MixedOp(args)
        # self.ffn_en = FFN1(d_ffn, dim_model, dropout=dropout_FFN)
        self.Local = MixedOpLocal(args)
        self.layer_norm_local = nn.LayerNorm(dim_model)
        # #out = out + self.drop_path_1 (skip_out) # here dropout is used for drop_path
        # #out = out + self.drop_path_2 (skip_out)
    def forward(self,out,NAS_coding):
        # skip_out = out  # same as ViT (pre-Norm)
        out = self.layer_norm1(out)
        skip_out = out
        out = self.atten(out,NAS_coding[0])
        # out = self.atten(out)
        out = out + skip_out
        #--------------
        out1 = self.Local(skip_out,NAS_coding=NAS_coding[2])
        # out1 = self.layer_norm_local(out1)
        out1 = self.layer_norm_local(out1+skip_out)
        #--------------
        # skip_out = out
        out = self.layer_norm2(out)
        skip_out = out
        # out = self.ffn_en(out,NAS_coding[1])
        out = self.ffn_en(out+out1,NAS_coding[1])
        # out = self.ffn_en(torch.cat([out,out1],dim=-1))
        # out = self.ffn_en(out)
        out = out + skip_out
        return out

class Decoder_Block(nn.Module):
    def __init__(self, dim_model, heads_de, seq_len, d_ffn=None, dropout_MHA=0.0, dropout_FFN=0.0, drop_path=0.0):
        super(Decoder_Block, self).__init__()
        if d_ffn is None:
            d_ffn = 4 * dim_model
            d_ffn = dim_model

        args = {'d_ffn': d_ffn,
                'dim_model':dim_model,
                'heads_en':heads_de,
                'seq_len':seq_len,
                'dropout_MHA':dropout_MHA,
                'dropout_FFN':dropout_FFN,
                }
        self.layer_norm1 = nn.LayerNorm(dim_model)
        # self.layer_norm1 = Identity()
        self.layer_norm2 = nn.LayerNorm(dim_model)
        # self.layer_norm2 = Identity()
        self.layer_norm3 = nn.LayerNorm(dim_model)
        # self.layer_norm3 = Identity()

        # last norm is defined before classifier
        self.atten_2 = SelfAttention(dim_model,heads=heads_de,dropout_rate= dropout_MHA,max_length=seq_len)


        # self.atten_1 = SelfAttention(dim_model,heads=heads_de,dropout_rate= dropout_MHA,max_length=seq_len)
        # self.atten_1 = MixedAttenConvOp(args)
        # self.ffn_de = FFN(d_ffn, dim_model, dropout=dropout_FFN)


        self.atten_1 = MixedOp(args)
        self.ffn_de = MixedOp(args)

        self.Local = MixedOpLocal(args)
        self.layer_norm_local = nn.LayerNorm(dim_model)
    def forward(self,out,encoder_out,NAS_coding):

        # skip_out = out
        out = self.layer_norm1(out)
        skip_out = out
        out = self.atten_1(out,NAS_coding[0])
        # out = self.atten_1(out)
        out = out + skip_out

        #----------------------
        out1 = self.Local(skip_out,NAS_coding=NAS_coding[2])
        # out1 = self.layer_norm_local(out1)
        out1 = self.layer_norm_local(out1+skip_out)

        #----------------------

        # skip_out = out
        out = self.layer_norm2(out)
        skip_out = out
        out = self.atten_2(out,encoder_out)
        # out = self.atten_2(out+out1,encoder_out)
        out = out + skip_out

        # skip_out = out       # same as ViT
        out = self.layer_norm3(out)
        skip_out = out
        # out = self.ffn_de(out,NAS_coding[1])
        out = self.ffn_de(out+out1 ,NAS_coding[1])
        # out = self.ffn_de(out)
        out = out + skip_out
        return out



