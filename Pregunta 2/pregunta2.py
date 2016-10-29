# coding=UTF-8
import urllib
import pandas as pd
import numpy as np
import re, time
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer, word_tokenize, data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#comentar a siguiente linea si se tienen instalado en el path por defecto
data.path.append('nltk_data')

# A
# cargar datos
train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
ftr = open("train_data.csv", "r")
fts = open("test_data.csv", "r")

# separar lineas en dos columnas: sentimiento y texto u opinion
rows = [line.split(" ",1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
train_df['Sentiment'] = pd.to_numeric(train_df['Sentiment'])
rows = [line.split(" ",1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
test_df['Sentiment'] = pd.to_numeric(test_df['Sentiment'])

print "Cantidad de registros train set ",train_df.shape
print "Cantidad de registros test set ",test_df.shape


# B
def word_extractor(text,sw=True,stmmng=True):
    porterstemmer = PorterStemmer()
    commonwords = stopwords.words('english')
    text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
    words = ""
    if stmmng:
        wordtokens = [ porterstemmer.stem(word.lower()) for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
    else:
        wordtokens = [ word.lower() for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
    for word in wordtokens:
        if sw:
            if word not in commonwords:
                words+=" "+word
        else:
            words+=" "+word
    return words

print "Stemming"
print word_extractor("I love to eat cake denied")
print word_extractor("I love eating cake")
print word_extractor("I loved eating the cake")
print word_extractor("I do not love eating cake")
print word_extractor("I don't love eating cake")
print word_extractor("You died last WeEk. It was sensational")
print word_extractor("owned cats, stemming cakes, factionally running") 

print "\nSin Stemming"
print word_extractor("I love to eat cake denied",stmmng=False)
print word_extractor("I love eating cake",stmmng=False)
print word_extractor("I loved eating the cake",stmmng=False)
print word_extractor("I do not love eating cake",stmmng=False)
print word_extractor("I don't love eating cake",stmmng=False)
print word_extractor("You died last WeEk. It was sensational",stmmng=False)
print word_extractor("owned cats, stemming cakes, factionally running",stmmng=False) 


# C
def word_extractor2(text,sw):
    wordlemmatizer = WordNetLemmatizer()
    commonwords = stopwords.words('english')
    text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
    words = ""
    wordtokens = [ wordlemmatizer.lemmatize(word.lower()) for word in word_tokenize(text.decode('utf-8','ignore')) ]
    for word in wordtokens:
        if sw:
            if word not in commonwords:
                words+=" "+word
        else:
            words+=" "+word
    return words

print "\nLematizador"
print word_extractor2("I love to eat cake denied",True)
print word_extractor2("I love eating cake",True)
print word_extractor2("I loved eating the cake",True)
print word_extractor2("I do not love eating cake",True)
print word_extractor2("I don't love eating cake",True)
print word_extractor2("You died last WeEk. It was sensational",True)
print word_extractor2("owned cats, stemming cakes, factionally running",True)

# D
def vector_rep(method,sw=True):
    if method == "stemming":
        if sw:
            texts_train    = [word_extractor(text) for text in train_df.Text]
            texts_test     = [word_extractor(text) for text in test_df.Text] 
        else:
            texts_train    = [word_extractor(text,sw=False) for text in train_df.Text]
            texts_test     = [word_extractor(text,sw=False) for text in test_df.Text]  
    elif method == "lemmatisation":
        if sw:
            texts_train    = [word_extractor2(text,True) for text in train_df.Text]
            texts_test     = [word_extractor2(text,True) for text in test_df.Text]
        else:
            texts_train    = [word_extractor2(text,False) for text in train_df.Text]
            texts_test     = [word_extractor2(text,False) for text in test_df.Text]

    vectorizer     = CountVectorizer(ngram_range=(1, 1), binary='False')
    vectorizer.fit(np.asarray(texts_train))
    features_train = vectorizer.transform(texts_train)
    features_test  = vectorizer.transform(texts_test)
    labels_train   = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
    labels_test    = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
    vocab          = vectorizer.get_feature_names()
    dist_train     = list(np.array(features_train.sum(axis=0)).reshape(-1,))
    return features_train, features_test, labels_train, labels_test, vocab, dist_train 


features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep("lemmatisation")
print "Set de entrenamiento: ",features_train.shape
print "Set de pruebas: ", features_test.shape
train_tags = []
test_tags = []
dist_test = list(np.array(features_test.sum(axis=0)).reshape(-1,))

for tag, count in zip(vocab, dist_train):
    train_tags.append([count, tag])
for tag, count in zip(vocab, dist_test):
    test_tags.append([count, tag])

train_tags.sort(reverse=True)
test_tags.sort(reverse=True)
print "\n----\nPalabras más frecuentes en set de entrenamiento\nRepeticiones\tPalabra"
for i in range(10):
    print "%d\t\t%s" % (train_tags[i][0], train_tags[i][1])
print "\nPalabras más frecuentes en set de prueba\nRepeticiones\tPalabra"
for i in range(10):
    print "%d\t\t%s" % (test_tags[i][0], test_tags[i][1])

# E
def score_the_model(model,x,y,xt,yt,text):
    acc_tr = model.score(x,y)
    acc_test = model.score(xt[:-1],yt[:-1])
    print "Training Accuracy %s: %f"%(text,acc_tr)
    print "Test Accuracy %s: %f"%(text,acc_test)
    print "Detailed Analysis Testing Results ..."
    print (classification_report(yt, model.predict(xt), target_names=['+','-']))
    #computar precisión recall fscore y soporte para cada clase
    precision, recall, fscore, support = precision_recall_fscore_support(yt, model.predict(xt))

    if text == 'BernoulliNB':
        #BNB = []
        BNB.append((acc_tr,acc_test,np.mean(precision),np.mean(recall),np.mean(fscore)))
    elif text == 'MULTINOMIAL':
        #multinomial = []
        multinomial.append((acc_tr,acc_test,np.mean(precision),np.mean(recall),np.mean(fscore)))
    elif text == 'LOGISTIC':
        #logistic = []
        logistic.append((acc_tr,acc_test,np.mean(precision),np.mean(recall),np.mean(fscore)))
    elif text == 'SVM':
        #svm = []
        svm.append((acc_tr,acc_test,np.mean(precision),np.mean(recall),np.mean(fscore)))

# F
def do_NAIVE_BAYES(x,y,xt,yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_the_model(model,x,y,xt,yt,"BernoulliNB")
    return model

BNB = []
print "Naive Bayes - BernoulliNB"
print "Lematizador con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation')
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)

print "Lematizador sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation',sw=False)
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)

print "Stemming con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming')
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)

print "Stemming con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming',sw=False)
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)

print "Conjunto aleatorio de prueba"
spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text

# G
def do_MULTINOMIAL(x,y,xt,yt):
    model = MultinomialNB()
    model = model.fit(x, y)
    score_the_model(model,x,y,xt,yt,"MULTINOMIAL")
    return model

multinomial = []
print "BI Multinomial"
print "Lematizador con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation')
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)

print "Lematizador sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation',sw=False)
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)

print "Stemming con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming')
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)

print "Stemming sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming',sw=False)
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)

print "Conjunto aleatorio de prueba"
spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text

# H
def do_LOGIT(x,y,xt,yt):
    start_t = time.time()
    Cs = [0.01,0.1,10,100,1000]
    for C in Cs:
        print "Usando C= %f"%C
        model = LogisticRegression(penalty='l2',C=C)
        model = model.fit(x, y)
        score_the_model(model,x,y,xt,yt,"LOGISTIC")
    return model

logistic = []
print "Regresión Logística regularizada"
print "Lematizador con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation')
model = do_LOGIT(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)

print "Lematizador sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation',sw=False)
model=do_LOGIT(features_train,labels_train,features_test,labels_test)

print "Stemming con StopWords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming')
model=do_LOGIT(features_train,labels_train,features_test,labels_test)

print "Stemming sin StopWords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming',sw=False)
model=do_LOGIT(features_train,labels_train,features_test,labels_test)

print "Conjunto aleatorio de prueba"
spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text

# I
def do_SVM(x,y,xt,yt):
    Cs = [0.01,0.1,10,100,1000]
    for C in Cs:
        print "El valor de C que se esta probando: %f"%C
        model = LinearSVC(C=C)
        model = model.fit(x, y)
        score_the_model(model,x,y,xt,yt,"SVM")
    return model

svm = []
print "Maquina de vectores de soporte"
print "Lematizador con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation')        
model = do_SVM(features_train,labels_train,features_test,labels_test)
test_pred = model.predict(features_test)

print "Lematizador sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('lemmatisation',sw=False)
model=do_SVM(features_train,labels_train,features_test,labels_test)

print "Stemming con Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming')
model=do_SVM(features_train,labels_train,features_test,labels_test)

print "Stemming sin Stopwords"
features_train, features_test, labels_train, labels_test, vocab, dist_train = vector_rep('stemming',sw=False)
model=do_SVM(features_train,labels_train,features_test,labels_test)

print "Conjunto aleatorio de prueba"
spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text

# j 
print "Comparaciones entre los modelos"