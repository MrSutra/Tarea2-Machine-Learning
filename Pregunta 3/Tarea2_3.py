  # -*- coding: latin-1 -*-
import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from pytagcloud import create_tag_image, make_tags
#from pytagcloud.lang.counter import get_tag_counts
from pytagcloud.colors import COLOR_SCHEMES
import seaborn as sns
import numpy as np
import scipy
import sys
import re
from pytagcloud.lang.stopwords import StopWords
from operator import itemgetter
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

def get_sparse_matrix(file_name):
   user_skill = open(file_name, 'r')

   print("..creating sparse matrix...")
   sparse_matrix = np.zeros((7890,14544))

   for line in user_skill:
      line = line.split(':')
      user = int(line[0]) 
      skills = line[1].split(',')
      for skill in skills:
         sparse_matrix[user][int(skill)] = 1 

   user_skill.close()
   print("...creating csr sparse matrix...")
   #csr_sparse_matrix = csr_matrix(sparse_matrix)
   return sparse_matrix

def get_skill_dictionary(file_name):

   skill_dict = {}
   skill_id_file = open(file_name, 'r')

   for line in skill_id_file:
      line = line.split(':')
      skill_name = str(line[0]).decode("utf-8")
      skill_id = str(line[1]).decode("utf-8")
      skill_dict[skill_id] = skill_name

   skill_id_file.close()
   
   return skill_dict

def get_training_testing_set(sparse_matrix):
   training_set = sparse_matrix[:5523][:]
   testing_set =  sparse_matrix[5523:][:]

   return training_set, testing_set

def plot_skills(sparse_matrix):

   skill_id = get_skill_dictionary('skill_id')
   skills_y = np.sum( sparse_matrix , axis = 0)
   skills_x = np.arange(14544)
   count = 0

   y = []
   x = []
   labels = []

   for i in range(14544):
      if skills_y[i] >= 550 :
         count += 1
         labels.append(skill_id[str(skills_x[i]) +'\n'].encode("utf-8"))
         y.append(skills_y[i])

   x = np.arange(count)     
   plt.barh(x, y,color="#4CAF50")
   for l in range(0,len(labels)):
      labels[l] = labels[l].decode("utf-8")
   
   labels[8] = "Informática Industrial".decode("utf-8")

   plt.yticks(x, labels, rotation = 'horizontal')
   plt.title("Top 10-Skills",fontsize=22)
   plt.xlabel("Skill Frequency",fontsize= 16)
   plt.ylabel("Skill Name",fontsize= 16)
   plt.show()


def get_tag_counts(text):
    """
    Search tags in a given text. The language detection is based on stop lists.
    This implementation is inspired by https://github.com/jdf/cue.language. Thanks Jonathan Feinberg.
    """
    words = map(lambda x:x, re.findall(r"[\w']+", text, re.UNICODE))
    
    s = StopWords()     
    s.load_language(s.guess(words))
    
    counted = defaultdict(int)
    
    for word in words:
        if not s.is_stop_word(word) and len(word) > 1:
            counted[word] += 1
      
    return sorted(counted.iteritems(), key=itemgetter(1), reverse=True)


def get_word_cloud(sparse_matrix):
   skill_id = get_skill_dictionary('skill_id')
   skills_y = np.sum( sparse_matrix , axis = 0)
   skills_x = np.arange(14544)
   count = 0

   y = []
   x = []
   labels = []
   for i in range(14544):
      if skills_y[i] >=  900 :
         count += 1
         labels.append(skill_id[str(skills_x[i]) +'\n'].encode("utf-8"))
         y.append(skills_y[i])

   x = np.arange(count)     
   
   for l in range(0,len(labels)):
      labels[l] = labels[l].decode("utf-8")

   text = ""

   for i in range(0,len(labels)):
      label = labels[i].replace("Microsoft","",1)
      label = label.title().replace(" ","") + " "
      text = text +  label*int(y[i]) + " "

   tags = make_tags(get_tag_counts(text), maxsize = 60, colors = COLOR_SCHEMES['audacity'])
 
   create_tag_image(tags, 'cloud_large.png', size = (1400,1200), background=(0, 0, 0, 255))


def create_training_dataset(training_set,index):

   y = training_set[:,index]
   X_std_train = StandardScaler().fit_transform(training_set)
   X_std_train = np.delete(training_set,index,1)

   return X_std_train,y

def create_test_dataset(testing_set,index):

   y_test = testing_set[:,index]
   X_std_test = StandardScaler().fit_transform(testing_set)
   X_std_test = np.delete(testing_set,index,1)

   return X_std_test,y_test


def get_LDA_performance(X_training,y_training,X_testing,y_testing):
   lda_model = LDA()
   lda_model.fit(X_training , y_training)

   return lda_model.score(X_testing,y_testing)

def get_PCA_performance(X_training,y_training,X_testing,y_testing):
   pca_model = PCA(n_components= 2000)
   pca_model.fit(X_training , y_training)

   return pca_model.score(X_training,y_training)

def score_the_model(model,x,y,xt,yt,text):
    acc_tr = model.score(x,y)
    acc_test = model.score(xt,yt)
    print "Training Accuracy %s: %f"%(text,acc_tr)
    print "Test Accuracy %s: %f"%(text,acc_test)
    print "Detailed Analysis Testing Results ..."
    print(classification_report(yt, model.predict(xt), target_names=['0','1']))

def do_MULTINOMIAL(x,y,xt,yt,imp=1):
    multinomial_model = MultinomialNB()
    multinomial_model.fit(x, y)
    print multinomial_model.score(xt,yt)
    print classification_report(yt, multinomial_model.predict(xt), target_names=['0','1','2'])



def get_BernoulliNB_performance(X_std_train,y_train,X_std_test,y_test):
    bernoulli_model = BernoulliNB()
    bernoulli_model.fit(X_std_train , y_train)
    print "SCORE: " + str(bernoulli_model.score(X_std_test,y_test))
    print classification_report(y_train, bernoulli_model.predict(X_std_train), target_names=['0','1'])
    


#GET PLOT AND WORD TAG

#skill_id = get_skill_dictionary('skill_id')
#get_word_cloud(sparse_matrix)

sparse_matrix = get_sparse_matrix('user_skill')
#print type(sparse_matrix)
np.random.shuffle(sparse_matrix)
#training_set, testing_set = get_training_testing_set(sparse_matrix)

#plot_skills(sparse_matrix)
#training_set = training_set[:1000][:]
#testing_set = testing_set[:1000][:]

#X_std_train, y_train = create_training_dataset(training_set,55)
#X_std_test, y_test = create_test_dataset(testing_set,55)

#get_BernoulliNB_performance(X_std_train,y_train,X_std_test,y_test)

#print get_LDA_performance(X_std_train,y_train,X_std_test,y_test)
#print get_PCA_performance(X_std_train,y_train,X_std_test,y_test)

#do_MULTINOMIAL(X_std_train,y_train,X_std_test,y_test)
#print X_std_test

















