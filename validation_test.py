'model_save_dir/best_CNNText_plus_acc_2020-06-04-14-15-46-183303'

from fastNLP.io import model_io
import torch
from guojia_sent_classifier import Guojia_Classifier_Pipe,CNNText_plus
import warnings
from fastNLP import Tester,AccuracyMetric,ClassifyFPreRecMetric
import sys
from fastNLP.core import Const as C
id2country_dict ={0:'澳大利亚',1:'韩国',2:'美国',3:'欧洲',4:'日本'}
test_mode =True  #True =批量测试,False =单独测试几个句子
inner_MODE =100
if __name__ == '__main__':
    mode = inner_MODE
    if mode == 0:  # 这个才是最简单的方法
        reloaded_model = torch.load('model_save_dir/best_CNNText_plus_acc_2020-06-04-14-15-46-183303')
    elif mode == 1:
        reloaded_model = CNNText_plus(5)
        reloaded_dict = torch.load('model_save_dir/best_CNNText_plus_acc_2020-06-04-14-15-46-183303').state_dict()
        reloaded_model.load_state_dict(reloaded_dict)
    else:
        try:
            reloaded_model = model_io.ModelLoader.load_pytorch_model(
                'model_save_dir/best_CNNText_plus_acc_2020-06-04-14-15-46-183303')
        except Exception as e:
            warnings.warn(e.args)
            sys.exit(1)
    reloaded_model.cuda()
    reloaded_model.eval()
    ds_final,id2country_dict = Guojia_Classifier_Pipe().process_from_file(r'C:\Users\24839\Desktop\jupyter-code\国家语料集合')
    test_data_plus = ds_final.get_dataset('test')
    if test_mode:
        print('##'*10)
        print('test data_count is :',len(test_data_plus))
        print('##'*10)
        tester_plus =Tester(data=test_data_plus,model=reloaded_model,metrics=AccuracyMetric())
        tester_plus.test()
    else:
        count=inner_MODE
        final_test_data =test_data_plus[:inner_MODE]
        print('##' * 10)
        print('test data_count is :', len(final_test_data))
        print('##' * 10)
        tester_plus =Tester(data=final_test_data,model=reloaded_model,metrics=AccuracyMetric())
        pred_result =[]
        tester_plus.test(i_want_the_predict_result=True,predict_result_saver=pred_result)
        # with torch.no_grad():
        #     pred_result =reloaded_model.predict(torch.tensor(final_test_data.field_arrays[C.INPUT].content,requires_grad=False,device='cuda:0'))  # return {C.OUTPUT: predict}
        print(pred_result,type(pred_result),len(pred_result),'将tensor的预测值转为列表后:',sep='\n')
        pred_result =torch.cat([_['pred'] for _ in pred_result],dim=0)
        pred_result =pred_result.tolist()
        print(pred_result,type(pred_result),len(pred_result))
        for name,value in final_test_data.field_arrays.items():
            if name ==C.RAW_WORD:
                sentence =value
            if name ==C.TARGET:
                real_flag =value
        print('真实标签',[_ for _ in real_flag])
        for i,j,k in zip(sentence,real_flag,pred_result):
            print(i,'\tflag:',j,'\t实际国家标签：',id2country_dict[j],'\tpredict_flag',k,'\t预测国家标签：',id2country_dict[k])