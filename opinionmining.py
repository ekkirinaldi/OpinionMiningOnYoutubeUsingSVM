# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import pickle
import math
import string
import re
import os
import scipy as sp
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import time
from nltk.probability import FreqDist
from collections import Counter
import urllib 
import simplejson
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

key='Insert Ur Key'

#Load ID video
with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/video_ids.txt') as s:
    video_ids = s.readlines()
video_ids = [x.strip() for x in video_ids]

#parsing data json
counter=1
for ID in video_ids:
    url = "https://www.googleapis.com/youtube/v3/commentThreads?key="+str(key)+"&textFormat=plainText&part=snippet&videoId="+str(ID)+"&maxResults=100"
    response = urllib.urlopen(url)
    data = simplejson.loads(response.read())

    with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/parsed_comments/'+str(counter)+'c.json', 'w') as outfile:
        simplejson.dump(data, outfile)
        
    url = "https://www.googleapis.com/youtube/v3/videos?id="+str(ID)+"&key=AIzaSyB-1OPGGifLu9v6e5tH_vM9h6ulCbrwozU&part=snippet,contentDetails,statistics,status"
    response = urllib.urlopen(url)
    data = simplejson.loads(response.read())

    with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/parsed_comments/'+str(counter)+'v.json', 'w') as outfile:
        simplejson.dump(data, outfile)
    
    counter=counter+1

#%%
#memanggil json yang telah disimpan
path_to_json = 'C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/parsed_comments'
json_comments = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('c.json')]
json_videos = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('v.json')]

df_raw = pd.DataFrame(columns=['Judul','Kanal','Deskripsi'])

for index, js in enumerate(json_videos):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_vids = simplejson.load(json_file)
        
        title=json_vids['items'][0]['snippet']['title']
        description=json_vids['items'][0]['snippet']['description']
        channelTitle=json_vids['items'][0]['snippet']['channelTitle']
        
        df_raw.loc[index] = [title,description,channelTitle]

content=[]
for index, js in enumerate(json_comments):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_comm = simplejson.load(json_file)
        
        comments=[item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in json_comm['items']]
        content.append(comments)
        
df_raw['comments']=content

#expand list komentar dalam dataframe_raw
komentar = df_raw['comments'].apply(pd.Series)
komentar = komentar.rename(columns = lambda x: 'komentar ke '+str(x))

#expand list label dalam dataframe_raw
labs = df_raw['labels'].apply(pd.Series)
labs = labs.rename(columns = lambda x: 'label ke '+str(x))
labs['idx'] = range(1, len(labs) + 1)
dflab = pd.melt(labs, id_vars=["idx"], var_name="Lab", value_name="Label")
dflab = dflab.drop(["idx","Lab"], axis=1)

df_raw = pd.concat([df_raw[:],komentar[:]], axis=1)
komen = [c for c in df_raw if c.startswith('komentar ke')]
df_raw = pd.melt(df_raw, id_vars=["Judul", "Kanal","Deskripsi","comments","labels"], var_name="Kom", value_name="Komentar")
df_raw = df_raw.drop(["comments","labels","Kom"], axis=1)
df_raw = pd.concat([df_raw[:],dflab[:]], axis=1)
df_raw = df_raw.dropna()

#%%
content=[]
for lab in df_raw['Label']:
    if 'product' in  lab:
        types='product'
    elif 'video' in lab:
        types='video'
    else:
        types='uninformative'
    content.append(types)
df_raw['Type']=content

content=[]
for lab in df_raw['Label']:
    if 'positive' in  lab:
        types='positive'
    elif 'negative' in lab:
        types='negative'
    else:
        types='neutral'
    content.append(types)
df_raw['Sentiment']=content

labels=[]
for label in df_raw['Label']:
    if label=='off-topic':
        lab='uninformative'
    elif label=='spam':
        lab='uninformative'
    else:
        lab=label
    labels.append(lab)
df_raw['All']=labels  

df_raw = df_raw.sort_values(["Type", "Judul"])
df_raw = df_raw.reset_index(drop=True)
processed = df_raw.copy() 

#dataframe siap
#%%
data=processed['Sentences'][:3]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
features = tfidf_vectorizer.get_feature_names()
indices = zip(*tfidf_matrix.nonzero())
for row,column in indices:
    print('(%d, %s) %f' %(row, features[column], data[row, column]))
    
#df_train=processed[6838:]
#print df_train.groupby('Label').count()
#%%
        
#unicoding
#membuat duplikat dataframe
content=[]
for comment in processed['Komentar']:
    c = re.sub(r'[^\x00-\x7f]',r' ',comment)
    content.append(c)
processed['Emoji_removal']=content


#Penghilangan garis miring (slash)
content=[]
for comment in processed['Emoji_removal']:
    strps=re.sub(r'/',r' ',comment)
    content.append(strps)
processed['Slashed']=content

#Spacing
content=[]
for comment in processed['Slashed']:
    content.append(re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", comment))
processed['Spaced']=content

#Slangword

#kamus slangword
swlist = eval(open("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/slangwords.txt").read())
pattern = re.compile(r'\b(' + '|'.join(swlist.keys()) + r')\b')

content =[]
for comment in processed['Spaced']:
    filteredSlang = pattern.sub(lambda x: swlist[x.group()], comment)
    content.append(filteredSlang.lower())
processed['Slangword']=content

#Akurasi keseluruhan pengujian TYPE kernel linear =  0.702421129861

#%%
#postag
folder = "D:/ta/pretag/"
foldes = "D:/ta/tagged/"

#%%   
os.chdir("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/IPOSTAgger_v1.1")

command = "java ipostagger C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/tests.txt"
command = command + " 1 1 0 1 > C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/tested.txt"
os.system(command)

#with open (fname) as f:
#    content = f.readlines()
#content = [x.strip() for x in content]
#%%
#simpan jadi file
file_number=1
for comment in processed['Slangword']:
    f = open(folder+"data"+str(file_number)+".txt", "w")
    f.write( str("".join(comment.lstrip(' ').splitlines()))  )      # str() converts to string
    f.close()
    file_number=file_number+1
    
os.chdir("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/IPOSTAgger_v1.1")

n = len([name for name in os.listdir(folder)])

file_number=0
for i in range(n): #ganti jumlah folder di
    file_number = str(i+1)
    command = "java ipostagger " + folder +"data"+ file_number +".txt"
    command = command + " 1 1 0 1 > " + foldes +"tagged"+ file_number +".txt"
    os.system(command)

n_tagged = len([name for name in os.listdir(foldes)])
content=[]
file_number=1
for i in range(n_tagged):
    file_object  = open(foldes+"tagged"+str(file_number)+".txt","r")
    content.append(file_object.readline())
    file_number=file_number+1

processed['Tagged']=content

#%%
##tokenized
content=[]
for comment in processed['Tagged']:
    content.append(WhitespaceTokenizer().tokenize(comment))
    
processed['Tokenized']=content

#tupling
tupled=[]
content=[]
for comment in processed['Tokenized']:
    tupled=[tuple(filter(None, x.split('/'))) for x in comment]
    content.append(tupled)
processed['Tuples']=content

#Unused class removal
classes=['CP','OP','...','.',',',':','CDP']
content=[]
for comment in processed['Tuples']:
    strps=[y for y in comment if y[1] not in classes ]
    content.append(strps)
 
processed['Classed']=content

#%%
#Penghilangan Tanda Baca (PUNCTUATION REMOVAL)
content=[]
for comment in processed['Classed']:
    strps=[y for y in comment if 'NEG' in y[1]]
    strps=len(strps)
    content.append(strps)
 
processed['Negation']=content
processed['Negation']=np.array(list(processed['Negation']), dtype=np.float)

#%%

#chunking

#product names
with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/mobilephonename_lists.txt') as p:
    phone_list = p.readlines()
phone_list = [x.strip().lower() for x in phone_list]  

#product/video tag
with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/productfeats.txt') as p:
    product_feats = p.readlines()
product_feats = [x.strip().lower() for x in product_feats]

with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/videofeats.txt') as p:
    video_feats = p.readlines()
video_feats = [x.strip().lower() for x in video_feats]

#product tagging
#%%
content=[]
for judul,comment in zip(processed['Judul'],processed['Classed']):
    for phone in phone_list:
        if phone.lower() in judul.lower():
            merek=phone
    with open('C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/files/phone_type/'+merek+'.txt') as p:
        types = p.readlines()
    types = [x.strip().lower() for x in types]
    model = [x.lower() for x in types if x.lower() in judul.lower()]
    model = [words for segments in model for words in segments.split()]

    product_related = list(product_feats)        
    product_related.append(merek)
    if len(model) > 0:
        product_related = product_related+model

    tupled=[]
    for i in comment:
        i=list(i)
        if i[0] in product_related:
            i[1]='PRODUCT'
        if i[0] in video_feats:
            i[1]='VIDEO'
        i=tuple(i)
        tupled.append(i)
    content.append(tupled)

#print content
processed['VP_Tagged']=content

#%%
#Penghilangan Tanda Baca (PUNCTUATION REMOVAL)
punch="!#$%&':(*+,-./;<=>?@[\])^_`{|}~"
content=[]
for comment in processed['VP_Tagged']:
    strps=[y for y in comment if y[0] and y[1] not in punch ]
    content.append(strps)
 
processed['Stripped']=content

#%%
#chunking
from nltk import RegexpParser
 
chunkToExtract = """
VP: {<PRODUCT><.*>?<NN>}
    {<PRODUCT><.*>?<JJ>}
    {<PRODUCT><SC><NN><.*>?}
    {<VIDEO><.*>?}
    """
parser = RegexpParser(chunkToExtract)

contentz=[]
for comment in processed['Stripped']:
    x=[z for z in comment if 'PRODUCT' in z[1]]
    y=[z for z in comment if 'VIDEO' in z[1]]    
    t=comment
    if len(x)>0 or len(y)>0:
        result = parser.parse(comment)
        for subtree in result.subtrees():
            if subtree.label() == 'VP':
                t = subtree.leaves()
    contentz.append(t)    
processed['Chunked']=contentz

#%%
content=[]
for comment in processed['Stripped']:
    sentence=[''.join(x[0]) for x in comment]
    content.append(' '.join(sentence))
    
processed['Sentences']=content 

#%%
#Cosine similarity

tfidf_vectorizer = TfidfVectorizer()
content=[]
for title,comment in zip(processed['Judul'],processed['Sentences']):
    doc=[]
    doc.append(title.lower())
    doc.append(comment.lower())
    doc_matrix = tfidf_vectorizer.fit_transform(doc)
    cc=cosine_similarity(doc_matrix[0], doc_matrix[1])
    content.append(cc[0].tolist())
    
processed['Similarity']=content 
processed['Similarity']=np.array(list(processed['Similarity']), dtype=np.float)

#%%
#    classifier = OneVsRestClassifier(SVC(kernel="linear",C=1.0, degree=3, 
#    shrinking=True, , tol=0.001, cache_size=200, class_weight=None, decision_function_shape=None,
#    verbose=False, random_state=None))


#FVEC
#K FOLD CROSS VALIDATION
processed = processed.sort_values(["Judul", "Kanal"])
processed = processed.reset_index(drop=True)

accs=[]
num_folds = 10
subset_size = len(processed.index)/num_folds
for i in range(num_folds):
    testing_this_round = processed[i*subset_size:][:subset_size]
    training_this_round = processed[:i*subset_size].append(processed[(i+1)*subset_size:], ignore_index=True)
    
    #train
    count_vectorizer = CountVectorizer(ngram_range=(1,2))
    train_vec = count_vectorizer.fit_transform(training_this_round['Sentences'])
    pickle.dump(count_vectorizer.vocabulary_,open("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/feature_train.pkl","wb")) 
    tfidf_transformer = TfidfTransformer(norm="l2",use_idf=True,smooth_idf=True)
    tfidf_transformer.fit(train_vec)
    tfidf_matrix = tfidf_transformer.transform(train_vec)
#    final_train = sp.sparse.hstack((tfidf_matrix, training_this_round[['Similarity','Negation']]))
    
    #test
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/feature_train.pkl", "rb")),ngram_range=(1,2))
    tfidf_test = tfidf_transformer.fit_transform(loaded_vec.fit_transform(testing_this_round['Sentences']))
#    final_test = sp.sparse.hstack((tfidf_test, testing_this_round[['Similarity','Negation']]))
    
    classifier = OneVsRestClassifier(SVC(kernel="poly",C=1.0, shrinking=True,
    tol=0.001, cache_size=200, class_weight=None, decision_function_shape=None,
    verbose=False, random_state=None, degree=2))

    classifier.fit(tfidf_matrix, training_this_round['Sentiment'])
    predicted = classifier.predict(tfidf_test)
    
    acc = accuracy_score(testing_this_round['Sentiment'], predicted)
    accs.append(acc)
    print "Akurasi fold ke "+ str(i+1)," = ", acc

print "Akurasi keseluruhan pengujian SENTIMENT kernel poly(2) = ",np.mean(accs)

#%%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
processed = processed.sort_values(["Type", "Judul"])

#train
traindata=processed[:2900]
traindata=traindata.append(processed[5760:8729])
traindata=traindata.append(processed[11698:12667])
traindata=traindata.reset_index(drop=True)  

#test
testdata=processed[2900:5760]
testdata=testdata.append(processed[8729:11698])
testdata=testdata.append(processed[12667:])
testdata=testdata.reset_index(drop=True)  

count_vectorizer = CountVectorizer(ngram_range=(1,2))
train_vec = count_vectorizer.fit_transform(traindata['Sentences'])
pickle.dump(count_vectorizer.vocabulary_,open("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/feature_train.pkl","wb"))
    
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(train_vec)
tfidf_matrix = tfidf_transformer.transform(train_vec)
final_train = sp.sparse.hstack((tfidf_matrix, traindata[['Similarity','Negation']]))

loaded_vec = CountVectorizer(vocabulary=pickle.load(open("C:/Users/Einherjar/Google_Drive/Tugas_Akhir/program/feature_train.pkl", "rb")),ngram_range=(1,2))
tfidf_test = tfidf_transformer.fit_transform(loaded_vec.fit_transform(testdata['Sentences']))
final_test = sp.sparse.hstack((tfidf_test, testdata[['Similarity','Negation']]))

classifier = OneVsRestClassifier(SVC(kernel="linear", shrinking=True, 
probability=False, tol=0.001, cache_size=200, class_weight=None, 
verbose=False, random_state=None, C=1.0))

classifier.fit(final_train, traindata['Sentiment'])
predicteds = classifier.predict(final_test)

classifier.fit(final_train, traindata['Type'])
predictedt = classifier.predict(final_test)

classifier.fit(final_train, traindata['All'])
predicteda = classifier.predict(final_test)

print classification_report(testdata['Sentiment'], predicteds)
print classification_report(testdata['Type'], predictedt)
print classification_report(testdata['All'], predicteda)

#%%


#expecteds = testdata['Sentiment']
#expectedt = testdata['Type']
#expecteda = testdata['All'] 

accs = accuracy_score(testdata['Sentiment'], predicteds)
acct = accuracy_score(testdata['Type'], predictedt)
acca = accuracy_score(testdata['All'], predicteda)

print 'Akurasi kernel rbf SENTIMENT = ', accs
print 'Akurasi kernel rbf TYPE      = ', acct
print 'Akurasi kernel rbf ALL       = ', acca
