r"""
.. todo::
    doc
"""

__all__ = [
    "CNNText"
]

import torch
import torch.nn as nn

from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..embeddings import embedding
from ..modules import encoder
from pytorch_pretrained_bert import BertModel  # from  fastNLP.modules import  BertModel 和pytorch_pretrained_bert官方包提供的机制一样
# ,BertTokenizer bert_tokenizer 移出去

class CNNText(torch.nn.Module):
    r"""
    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'
    
    """

    def __init__(self, embed,
                 num_classes,
                 input_dims =None,
                 bert_model:BertModel =None,
                # bert_tokenizer 移出去
                 kernel_nums=(30, 40, 50),
                 kernel_sizes=(1, 3, 5),
                 dropout=0.5):
        r"""
        embed 必填，当为None时，采用bert输入，非None时，采用embdding输入
        input_dims&bert_model&bert_tokenizer 是利用bert输入的形参，默认为None->不启用bert
        
        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param int num_classes: 一共有多少类
        :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
        :param float dropout: Dropout的大小
        """
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        if embed ==None:
            self.embed =None
            self.bert_model =bert_model
            # self.bert_tokenizer =bert_tokenizer  bert_tokenizer 移出去
            self.conv_pool=encoder.ConvMaxpool(
                in_channels=input_dims,
                out_channels=kernel_nums,
                kernel_sizes=kernel_sizes
            )
        else:
            self.embed = embedding.Embedding(embed)
            self.conv_pool = encoder.ConvMaxpool(
                in_channels=self.embed.embedding_dim,
                out_channels=kernel_nums,
                kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, words, seq_len=None):
        r"""
        :param (str | list for words) words: 一个整句或者句子的字列表
        XXX:param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        if self.embed ==None:
            # words_bert_id=self.bert_tokenizer.convert_tokens_to_ids(['CLS']+list(words)+['SEP'])
            with torch.no_grad(): # TODO
                words_bert_id =words
                words_embed,_ = self.bert_model(words_bert_id,output_all_encoded_layers =False)
            x =words_embed
        else:
            x = self.embed(words)  # [N,L] -> [N,L,C]

        if seq_len is not None:
            mask = seq_len_to_mask(seq_len)
            x = self.conv_pool(x, mask)
        else:
            x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {C.OUTPUT: x}

    def predict(self, words, seq_len=None):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
