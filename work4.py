import utils
import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_data():
    seg_list = utils.singleword()
    seglist = ''
    for i in seg_list:
        seglist += i
        seglist += ' '
    with open('spilt.txt','w',encoding='utf-8') as f:
        f.write(seglist)
    return seg_list

def train_vec(file = "spilt.txt"):
    data = open(file, 'r',encoding='utf-8').read().split('\n')
    model = Word2Vec(data,min_count=1,workers=6,iter=6,window=20)
    char_to_index = {ch: i for i, ch in enumerate(model.wv.index2word)}
    pickle.dump([model.syn1neg,model.wv.index2word,char_to_index],open("vec_params.pkl",'wb'))
    return [model.syn1neg,model.wv.index2word,char_to_index]

def getcontent(seq_length,file = "spilt.txt"):
    data = open(file, 'r', encoding='utf-8').read().split(' ')
    datas = []
    for i in range(0, len(data) - seq_length, seq_length):
        datasplit = data[i:i + seq_length ]
        text = "".join(datasplit)
        datas.append((text))
    return datas

class MyDataset(Dataset):
    #加载数据
    def __init__(self,data,w1,char_to_index):
        self.data = data
        self.w1 = w1
        self.char_to_index = char_to_index

    #获取数据并处理
    def __getitem__(self, index):
        words = self.data[index]
        words_index = [self.char_to_index[word] for word in words]
        xs_index = words_index[:-1]
        ys_index = words_index[1:]
        xs_embedding = self.w1[xs_index]
        #print(xs_index,index)
        return xs_embedding,np.array(ys_index).astype(np.int64)

    #获取数据总长度
    def __len__(self):
        return len(self.data)

class LSTM(nn.Module):
    def __init__(self,embedding_num,hidden_num,word_size):
        super().__init__()
        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.word_size = word_size
        self.lstm = nn.LSTM(input_size = embedding_num,hidden_size = hidden_num,batch_first = True,num_layers=2)
        self.dropout = nn.Dropout(0.3) #随机失活率
        self.flatten = nn.Flatten(0,1)
        self.liner = nn.Linear(hidden_num,word_size)
        self.cross_entory = nn.CrossEntropyLoss()
    def forward(self,xs_embedding, h_0=None, c_0=None):
        if h_0 ==None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),np.float32))
            c_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),np.float32))
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        xs_embedding = xs_embedding.to(device)
        hidden,(h_0,c_0)= self.lstm(xs_embedding,(h_0, c_0))
        hidden_drop = self.dropout(hidden)
        flatten = self.flatten(hidden_drop)
        pre = self.liner(flatten)
        return pre,(h_0,c_0)

def generate(syn1neg,index_to_char,char_to_index,hidden_num):
    result = ''
    wordfirst = '在'
    word_index = char_to_index[wordfirst]
    result += wordfirst
    h_0 = torch.tensor(np.zeros((2, 1, hidden_num), np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hidden_num), np.float32))
    h_0 = h_0.to(device)
    c_0 = c_0.to(device)
    for i in range(200):
        word_embedding = syn1neg[word_index]
        word_embedding = torch.tensor(word_embedding.reshape(1,1,-1))
        pre, (h_0,c_0)= model(word_embedding,h_0,c_0)
        word_index = int(torch.argmax(pre))
        result += index_to_char[word_index]
    print(result)

def generate_poetry_auto():
    result = ""
    word_index = np.random.randint(0, word_size, 1)[0]

    result += index_to_char[word_index]
    h_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))

    for i in range(31):
        word_embedding = torch.tensor(syn1neg[word_index][None][None])
        pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        result += index_to_char[word_index]

    return result

if __name__ == "__main__":

    device = 'cuda'
    seg_list = get_data()
    [syn1neg,index_to_char,char_to_index] = train_vec()
    batch_size = 128
    seq_length = 200
    hidden_num = 128
    lr = 0.07
    epochs = 5000
    word_size, embedding_num = syn1neg.shape
    datas = getcontent(seq_length)
    dataset = MyDataset(datas,syn1neg,char_to_index)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    model = LSTM(embedding_num,hidden_num,word_size)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epochs):
        for batch_index,(xs_embedding,ys_index) in enumerate(dataloader):
            model.train()
            xs_embedding = xs_embedding.to(device)
            ys_index = ys_index.to(device)
            pre,_= model.forward(xs_embedding)
            loss = model.cross_entory(pre,ys_index.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"loss:{loss:.3f},e{e}")
        print(generate_poetry_auto())
        #generate(syn1neg, index_to_char, char_to_index, hidden_num)



