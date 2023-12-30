import tensorflow as tf 
import os
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from gensim.models import KeyedVectors
import numpy as np
import nltk
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 
text2array = BOW_500.build_tokenizer()
model_path = "../GoogleNews/GoogleNews-vectors-negative300.bin.gz"
w2v_google_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
def get_vector(x):
    '''
    prevent KeyError on KeyedVectors.getvector
    '''
    try:
        return w2v_google_model.get_vector(x,norm=True)
    except KeyError:
        x = np.ones(300,np.float32)
        return x / np.linalg.norm(x)
def _text2vector_google(x):
    '''
     return 300*300 vector per tweet
    '''
    x = x['text']
    l = len(x)
    if l==0:
        raise ValueError
    if l>300:
        x = x[:300]
    vector = map(get_vector,x)
    out = np.stack(list(vector))
    if l<300:
        t = np.zeros((300-l,300),np.float32)
        out = np.concatenate([out,t])
    assert out.shape == (300,300)
    return out
def _text2vetor_bow500(x):
    '''
    return 20*500 vector per tweet
    '''
    x = text2array(x['text'])
    l = len(x)
    if l==0:
        raise ValueError
    if l>300:
        x = x[:300]
    hug_vec = BOW_500.transform(x).toarray()
    if l<300:
        t = np.zeros((300-l,500),np.float32)
        hug_vec = np.concatenate([hug_vec,t])
    assert hug_vec.shape == (300,500)
    l = [np.sum(hug_vec[i:i+15],axis=0) for i in range(0,286,15)]
    out = np.concatenate(l)
    assert out.shape == (20,500)
    return out
def _check_saved():
    return False
    pass
def _check_raw():
    assert os.path.isfile('../kaggle-data/data_identification.csv')
    assert os.path.isfile('../kaggle-data/emotion.csv')
    assert os.path.isfile('../kaggle-data/sampleSubmission.csv')
    assert os.path.isfile('../kaggle-data/tweets_DM.json')
def _read_json():
    with open('../kaggle-data/tweets_DM.json') as f:
        data = json.load(f)
    return data
def data_to_df():
    '''
    create training & testing dataframe, with 'id', 'text', and 'emotion' for training. 
    '''
    if _check_saved():
        pass
    _check_raw()
    temp = pd.read_csv('../kaggle-data/emotion.csv')
    id2emtion = dict()
    foo = lambda x:id2emtion.setdefault(x[0],x[1])
    temp.apply(foo,axis=1)
    emotions = pd.unique(temp['emotion'])
    temp = pd.read_csv('../kaggle-data/data_identification.csv')
    id2test = dict()
    foo = lambda x:id2test.setdefault(x[0],x[1]=='test')
    temp.apply(foo,axis=1)
    
    array2token = BOW_500.build_preprocessor()
    ids, texts, text_emotions= [],[],[]
    test_ids,test_texts =[],[]
    for row in open('../kaggle-data/tweets_DM.json').readlines():
        
        temp = json.loads(row)['_source']['tweet']
        tweet_text = temp['text'].lower()
        tweet_text = tweet_text.replace('\r', '')
        tweet_text = tweet_text.replace('\n', '')
        id = temp['tweet_id']
        if id2test[id]:
            test_ids.append(id)
            test_texts.append(array2token(tweet_text))
        else:
            ids.append(id)
            texts.append(array2token(tweet_text))
            text_emotions.append(id2emtion[id])
    training_data_df = pd.DataFrame({'id':ids,'text':texts,'emotion':text_emotions})
    training_data_df.to_csv('../kaggle-data/my_training_data.csv')
    testing_data_df = pd.DataFrame({'id':test_ids,'text':test_texts})
    testing_data_df.to_csv('../kaggle-data/my_testing_data.csv')
    print('done')
def create_tf_datasets():
    '''
    create TF dataset form dataframe, return (list of dataset, testing set)
    '''
    if not os.path.isfile('../kaggle-data/my_training_data.csv') or \
        not os.path.isfile('../kaggle-data/my_training_data.csv'):
        print('no csv')
        data_to_df()

    raw_training_df = pd.read_csv('../kaggle-data/my_training_data.csv')
    raw_testing_df = pd.read_csv('../kaggle-data/my_training_data.csv')

    mlb = LabelBinarizer()
    mlb.fit(raw_training_df['emotion'])
    emotions = pd.unique(raw_training_df['emotion'])
    datasets = []
    weights = []
    for emotion in emotions:
        if not os.path.isdir('../kaggle-data/'+str(emotion)):
            emo_df = raw_training_df[raw_training_df.emotion == emotion]
            weights.append(len(emo_df))
            vectors = BOW_500.transform(emo_df['text']).toarray()
            temp = mlb.transform(emo_df['emotion'])
            semo_dataset = tf.data.Dataset.from_tensor_slices((vectors,temp))
            semo_dataset.save('../kaggle-data/{}'.format(emotion))
        else :
            semo_dataset = tf.data.Dataset.load('../kaggle-data/'+str(emotion))
        datasets.append(semo_dataset)
    if not os.path.isdir('../kaggle-data/testing'):
        vectors = BOW_500.transform(raw_testing_df['text']).toarray()
        test_dataset = tf.data.Dataset.from_tensor_slices(vectors)
    else:
        test_dataset = tf.data.Dataset.load('../kaggle-data/testing')
    return datasets,test_dataset
def load_dataset():
    '''
    load dataset from dir, return (list of dataset, testing set)
    '''
    raw_training_df = pd.read_csv('../kaggle-data/my_training_data.csv').dropna()
    emotions = pd.unique(raw_training_df['emotion'])
    test_dataset = tf.data.Dataset.load('../kaggle-data/testing')
    datasets,weights = [],[]
    for emotion in emotions:
        semo_dataset = tf.data.Dataset.load('../kaggle-data/'+str(emotion))
        datasets.append(semo_dataset)
    return datasets,test_dataset
if __name__ == '__main__':
    _check_raw()
    create_tf_datasets()