import tensorflow
import re
from bert_serving.client import BertClient
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import re


def split_zh_sentence(sentence):
    tokens = []
    tokens.append("[CLS] ")
    for word in sentence.strip():
        tokens.append(word)
        if word == u'。':
            tokens.append(" [SEP] ")
        if word == u'?':
            tokens.append(" [SEP] ")
    return tokens


def getPredicted(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    l = split_zh_sentence(sentence)
    tokenized_text = tokenizer.tokenize("".join(l))

    # replace the English token to Chinese
    nerver_split = ['CLS', 'UNK', 'SEP']
    masked_index = []
    for i in range(len(tokenized_text)):
        if re.findall(r'[a-zA-Z]+', tokenized_text[i]):
            e = re.findall(r'[a-zA-Z]+', tokenized_text[i])[0]
            if e not in nerver_split:
                tokenized_text[i] = '[MASK]'
                masked_index.append(i)

    print(masked_index)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    # confirm we were able to predict 'henson'
    predicted_token = []
    for i in masked_index:
        if i:
            predicted_index = torch.argmax(predictions[0, i]).item()
            predicted = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            predicted_token.append(predicted)
            tokenized_text[i] = predicted
        else:
            break

    return tokenized_text, predicted_token


def getEng(path):
    with open(path, 'r')as f:
        sentences = [line for line in f]

    sentiment = []
    text = []
    for i in sentences:
        text.append(i.split(',')[0].strip())
        sentiment.append(i.split(',')[1])

    count = 0
    useful = []
    for i in range(len(text)):
        count += 1
        if re.findall('[a-zA-Z]+', text[i]):
            useful.append(i)

    print('Chinese-English mixed text proportion %f'%(len(useful)/count))

    X, y = [], []
    for i in useful:
        X.append(text[i])
        y.append(sentiment[i])

    for i in range(len(y)):
        if y[i] == 'NEG':
            y[i] = 0
        if y[i] == 'negative':
            y[i] = 0
        if y[i] == 'POS':
            y[i] = 1
        if y[i] == 'positive':
            y[i] = 1
    return X, y
    # for i in range(len(sentence)):
    #     pattern = re.compile(u"...展开全文c\"")
    #     sentence[i] = re.sub(pattern, '', sentence[i]).split(' ?')[0]
    #     # print(sentence[i])
    #
    # result_sentence = []
    # my_re = re.compile(r'[A-Za-z]+')
    # count = 0
    # count_all = 0
    # for i in sentence:
    #     count_all += 1
    #     if len(re.findall(my_re, i)):
    #         result_list = re.findall('@[a-zA-Z]+', i)
    #         result_sentence.append(i)
    #         count += 1


if __name__ == '__main__':
    toremove = ['[CLS]', '[UNK]', '[SEP]']

    X, y = getEng('./original.txt')
    print(X)
    print(y)

    # w1 = open("./original.txt", 'w')
    # for i in range(len(X)):
    #     w1.write(''.join(X[i].split(' ')) + ',')
    #     w1.write(str(y[i]))
    #     w1.write('\n')

    w = open('./translation.txt', 'w')
    for i in range(len(X)):
        a, _ = getPredicted(X[i])
        l_pre = ''.join(a)
        for j in toremove:
            l_pre = l_pre.replace(j, '')
        w.write(l_pre + ',')
        w.write(str(y[i]))
        w.write('\n')


    '''
    w1 = open("./original.txt", 'w')
    for i in result_sentence:
        w1.write(i)
        w1.write('\n')
    '''
