import os
import re
from tqdm import tqdm
from collections import Counter
import jieba
import math
#import thulac
jieba.load_userdict("cn_stopwords.txt")
def getpath(origin_path):#获取全部txt文件
    names = os.listdir(origin_path)
    pathlist = []
    for name in tqdm(names):
        if name.lower().endswith('.txt'):
            path = os.path.join(origin_path, name)
            pathlist.append(path)
    return pathlist

def getcontext_nonoise(path):
    with open(path, "r", encoding="ANSI") as file:
        filecontext = file.read()
        filecontext = nonoise(filecontext)
        #seg_list = jieba.lcut_for_search(filecontext)
    return filecontext

def getcontext(path):
    with open(path, "r", encoding="ANSI") as file:
        filecontext = file.read()
        #seg_list = jieba.lcut_for_search(filecontext)
    return filecontext

def nonoise(filecontext):
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str1 = u'[②③④⑤⑥⑦⑧⑨⑩_“”、。《》！，：；？‘’」「…『』（）<>【】．·.—*-~﹏]'
    filecontext = re.sub(english, '', filecontext)
    filecontext = re.sub(str1, '', filecontext)
    filecontext = filecontext.replace("\n", '')
    filecontext = filecontext.replace(" ", '')
    filecontext = filecontext.replace(u'\u3000', '')
    return filecontext

def calc_tf(corpus):
    #   统计每个词出现的频率
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 1
        word_freq_dict[word] += 1
    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    # 计算TF概率
    word_tf = dict()
    # 信息熵
    shannoEnt = 0.0
    # 按照频率，从高到低，开始遍历，并未每个词构造一个id
    for word, freq in word_freq_dict:
        # 计算p(xi)
        prob = freq / len(corpus)
        word_tf[word] = prob
        shannoEnt -= prob * math.log(prob, 2)
    return word_freq_dict, shannoEnt

def singleword():
    origin_path = (r".\jyxstxtqj_downcc.com")
    pathlist = getpath(origin_path)
    content = []
    for path in pathlist:
        content += getcontext_nonoise(path)
    #word_tf, shannoEnt = calc_tf(content)
    #print(word_tf)
    #print(shannoEnt,len(content),len(word_tf))
    return content
    #content = no_noise(content)
    #print(content)

def ciku():
    origin_path = (r".\jyxstxtqj_downcc.com")
    pathlist = getpath(origin_path)
    content = ''
    for path in pathlist:
        content += getcontext(path)
    seg_list = jieba.lcut_for_search(content)
    '''thu1 = thulac.thulac(seg_only=True)  # 默认模式
    text = thu1.cut(content, text=True)  # 进行分词
    seg_list = text.split()'''
    for num in range(len(seg_list)):
        seg_list[num] = nonoise(seg_list[num])
    seg_list = [i for i in seg_list if i != '']
    #print(seg_list, len(seg_list))
    #word_tf, shannoEnt = calc_tf(seg_list)
    #print(word_tf[:10])
    #print(shannoEnt, len(seg_list), len(word_tf))
    return seg_list


if __name__ == "__main__":
    ciku()