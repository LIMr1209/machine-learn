from transformers import pipeline

# # Allocate a pipeline for sentiment-analysis
# classifier = pipeline('sentiment-analysis')
# a  = classifier('You are a fool.')
# print(a)
# [{'label': 'POSITIVE', 'score': 0.9978193640708923}]


import torch
from transformers import BertModel, BertTokenizer
# 这里我们调用bert-base模型，同时模型的词典经过小写处理
model_name = 'bert-base-chinese'
# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name,mirror='tuna')
# 载入模型
model = BertModel.from_pretrained(model_name)
# 输入文本
input_text = "我是一个傻逼"
# 通过tokenizer把文本变成 token_id
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
# tensor([[ 101, 2769, 3221,  671,  702, 1004, 6873, 1506, 1506,  102]])
input_ids = torch.tensor([input_ids])
print(input_ids)
# 获得BERT模型最后一个隐层结果
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
