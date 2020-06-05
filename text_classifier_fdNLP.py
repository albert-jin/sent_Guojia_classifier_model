from fastNLP.io import SST2Pipe
import spacy
from fastNLP.models import CNNText
from  fastNLP.models.cnn_text_classification import CNNText
# import en_core_web_sm as en
# en =spacy.load('en_core_web_sm')
pipe =SST2Pipe()
databundle =pipe.process_from_file(r'C:\Users\24839\.fastNLP\dataset\SST-2')
vocab =databundle.get_vocab('words')
print(databundle,databundle.get_dataset('train')[0],databundle.get_vocab('words'),sep='$$')
