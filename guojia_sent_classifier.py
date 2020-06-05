'''
C:\\Users\\24839\\Desktop\\jupyter-code\\国家语料集合
'''
from fastNLP.io import Pipe,Loader,DataBundle  # SST2Pipe,
from numpy import random
import os
from pathlib import Path
from fastNLP.core import  Instance,DataSet
import warnings
from typing import Union, Dict
from fastNLP.core.const import Const as C
# pipe =SST2Pipe()
# data =pipe.process_from_file(r'C:\Users\24839\.fastNLP\dataset\SST-2')

class Guojia_Classifier_Loader(Loader):
    '''
    从文件夹中读取所有txt文件，每个txt就是一个国家的分类，txt中每个文本占一行
    '''
    def __init__(self):
        super().__init__()
        pass


    def _load(self, path: str) -> tuple:
        '''
        :param path: 存储国家句子的文件路径 path
        :return: 返回 DataSet对象
        '''
        for root, dirs,filenames in os.walk(path,topdown=False):
            ds =DataSet()
            id2country_dict =dict()
            country_set ={'韩', '美', '欧', '日', '澳'}
            index =0
            for filename in filenames:
                country ='其他'
                if filename.find('.txt') != -1:
                    for country in country_set:
                        if country in filename:
                            id2country_dict.update({index:country})
                            break
                    file = os.path.join(root, filename)
                    warnings.warn('now found path:{} file,read it'.format(file))
                    count =0
                    with open(file,mode='r',encoding='utf-8') as inp:
                        for line in inp:
                            line =line.strip()
                            sent =line[:300]
                            if len(sent)>0:
                                count+=1
                                ds.append(Instance(raw_words =sent,target =index))
                            if count>5000:  #数据比例不合理，我将超出的数据5000之后的舍弃了
                                break
                        index +=1
                    warnings.warn('from path: {} file,read data(sentence) sum:{},the flag count: {}'.format(file,count,index))
            if len(ds) ==0:
                raise  Exception('读入失败，数据量为空，可能是读入的文件夹里面没有txt')
            return ds,id2country_dict
    def load(self, paths: Union[str, Dict[str, str]] = None ,ratio_train_dev_test:tuple =(8,1,1)) -> tuple:
        '''
        调用_load函数,对其return value做数据集划分的处理，(train,val),test
        :param paths:
        :return: DataBundle
        '''
        datasets,id2country_dict =self._load(paths)
        train_data =DataSet()
        dev_data =DataSet()
        test_data =DataSet()
        indices =[_ for _ in range(len(datasets))]
        random.shuffle(indices)
        train_count =int(len(datasets)*(ratio_train_dev_test[0]/sum(ratio_train_dev_test)))
        dev_count =int(len(datasets)*(ratio_train_dev_test[1]/sum(ratio_train_dev_test)))
        test_count =int(len(datasets)*(ratio_train_dev_test[2]/sum(ratio_train_dev_test)))
        train_indices =indices[:train_count]
        dev_indices =indices[train_count:train_count+dev_count]
        test_indices =indices[train_count+dev_count:]
        for idx in train_indices:
            train_data.append(datasets[idx])
        for idx in dev_indices:
            dev_data.append(datasets[idx])
        for idx in test_indices:
            test_data.append(datasets[idx])
        warnings.warn('分割train/dev/test集合，count:{}/{}/{}'.format(len(train_data),len(dev_data),len(test_data)))
        data_set ={'train':train_data,'dev':dev_data,'test':test_data}
        return DataBundle(datasets=data_set),id2country_dict

from pytorch_pretrained_bert import BertTokenizer
class Guojia_Classifier_Pipe(Pipe):
    def __init__(self):
        super().__init__()
        self.bert_tokenizer_real = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=r'C:\Users\24839\Desktop\科研\bert预训练模型\chinese_L-12_H-768_A-12')
    def copy_func(self, words):
        words_bert_id = self.bert_tokenizer_real.convert_tokens_to_ids(['CLS'] + list(words) + ['SEP'])
        return words_bert_id
    def process(self, data_bundle: DataBundle) -> DataBundle:
        data_bundle.copy_field(field_name=C.RAW_WORD, new_field_name=C.INPUT, ignore_miss_dataset=True)
        for name,dataset in data_bundle.datasets.items():
            dataset.apply_field(self.copy_func, field_name=C.RAW_WORD, new_field_name=C.INPUT)
            dataset.add_seq_len(C.INPUT)  # 这里没有用Const.INPUT=words而是 raw_words
        data_bundle.set_input(C.INPUT,C.INPUT_LEN)
        data_bundle.set_target(C.TARGET)  # Const.TARGET ,'target'
        return data_bundle
    def process_from_file(self, paths) -> tuple:
        data_bundles,id2country_dict = Guojia_Classifier_Loader().load(paths=paths)
        return self.process(data_bundle=data_bundles),id2country_dict

import torch
import torch.nn as nn
from fastNLP.models import CNNText

from fastNLP.core.const import Const as C
print(C.TARGET,C.INPUT,C.LOSS,C.INPUT_LEN,C.RAW_WORD)

from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import embedding
from fastNLP.modules import encoder

from pytorch_pretrained_bert import BertModel,BertConfig  # from  fastNLP.modules import  BertModel 和pytorch_pretrained_bert官方包提供的机制一样
bert_model_real =BertModel.from_pretrained(pretrained_model_name_or_path=r'C:\Users\24839\Desktop\科研\bert预训练模型\pytorch_bert_base_chinese')
config =BertConfig.from_json_file(r'C:\Users\24839\Desktop\科研\bert预训练模型\pytorch_bert_base_chinese\bert_config.json')

class CNNText_plus(CNNText):
    r"""
    基于
    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'

    更改了CNNText模型的初始化参数，变为bert+CNNText模型
    """

    def __init__(self,num_classes,embed=None,bert_model =bert_model_real,  #bert_tokenizer=bert_tokenier_real, \
                 input_dims =config.hidden_size,kernel_nums=(30, 40, 50),kernel_sizes=(1, 3, 5),dpot =0.5): # input_dims =768,
        r"""

        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param int num_classes: 一共有多少类
        :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
        :param float dropout: Dropout的大小
        :param bert_model&bert_tokenizer
        """
        super(CNNText_plus, self).__init__(embed =embed,
                 num_classes=num_classes,
                 bert_model=bert_model,
                 # bert_tokenizer=bert_tokenizer, bert_tokenizer 移出去
                 input_dims= input_dims,
                 kernel_nums=kernel_nums,
                 kernel_sizes=kernel_sizes,
                 dropout=dpot)
        self.cuda()

from fastNLP import AccuracyMetric,Const
from fastNLP import CrossEntropyLoss
import torch.optim as optim
from fastNLP import Trainer
if __name__ == '__main__':
    ds_final,id2country_dict = Guojia_Classifier_Pipe().process_from_file(r'C:\Users\24839\Desktop\jupyter-code\国家语料集合')
    print(ds_final.get_dataset('train')[:5],ds_final.get_dataset('train').print_field_meta(),sep='\n')
    print('##'*10)
    train_data_plus = ds_final.get_dataset('train')
    test_data_plus = ds_final.get_dataset('test')
    dev_data_plus = ds_final.get_dataset('dev')
    print(len(train_data_plus), len(test_data_plus), len(dev_data_plus))
    print('##'*10)
    model_cnntext_plus = CNNText_plus(5)
    print(model_cnntext_plus)
    metric_plus = AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)
    loss_plus = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)
    optimizer_plus = optim.RMSprop(model_cnntext_plus.parameters(), lr=0.005, alpha=0.99, eps=1e-8)  #原来是0.01
    N_EPOCHS = 5
    BATCH_SIZE = 50
    trainer_plus = Trainer(model=model_cnntext_plus, train_data=train_data_plus, dev_data=dev_data_plus, loss=loss_plus,
                           metrics=metric_plus, optimizer=optimizer_plus, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,save_path='model_save_dir')
    trainer_plus.train()