'''
Author: WEY
Date: 2021-05-13 10:18:46
LastEditTime: 2021-05-18 21:00:24
'''
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":
  
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  
  model = AutoModel.from_pretrained("bert-base-uncased")
  
  inputs = tokenizer("Hello world!", return_tensors="pt")
  
  outputs = model(**inputs)


# With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
