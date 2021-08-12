import json
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/datadrive/saved_data/transformers-models')
model_encode = RobertaModel.from_pretrained('roberta-base', cache_dir='/datadrive/saved_data/transformers-models')

def generate_embedding(texts):
    encoded_input = tokenizer(texts, return_tensors='pt')
    print(encoded_input)
    output = model_encode(**encoded_input)
    return output

if __name__=='__main__':

    text = "Replace"
    embed = generate_embedding(text)
    print(embed[0].shape)
    print(embed[1].shape)
    print(embed[0][0][0][0], embed[0][0][0][1], embed[0][0][0][2], embed[0][0][0][3])
    print(embed[1][0])
    print(len(embed))

