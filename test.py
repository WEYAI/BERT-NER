'''
Author: WEY
Date: 2021-05-13 10:18:46
LastEditTime: 2021-05-13 14:59:44
'''
import torch
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
        # INPUT TEXT MUST BE ALReady word-segmented!
        line = "tôi là sinh_viên trường đại_học công_nghệ ."
        phobert = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        # [1*9] B*T
        input_ids = torch.tensor([tokenizer.encode(line)])
        print(input_ids)
        # [1*9*768] B*T*h
        with torch.no_grad():
            features = phobert(input_ids)  # models outputs are now tuples
            print(features.shape())


# With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
