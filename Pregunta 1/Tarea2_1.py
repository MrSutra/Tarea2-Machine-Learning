import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scipy

train_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
test_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.test" 

def get_data(train_data_url,test_data_url):
   
   train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
   test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")

   train_df = pd.DataFrame.from_csv('train_data.csv',header=0,index_col=0)
   test_df = pd.DataFrame.from_csv('test_data.csv',header=0,index_col=0)

   train_df.head()
   test_df.tail()

   return train_df, test_df


def scale_data(train_df, test_df):
   X = train_df.ix[:,'x.1':'x.10'].values
   y = train_df.ix[:,'y'].values
   X_std = StandardScaler().fit_transform(X)

   return X_std, y


def get_PCA(X_std):
   sklearn_pca =  PCA(n_components = 2)
   Xred_pca = sklearn_pca.fit_transform(X_std)
   cmap = plt.cm.get_cmap('RdYlBu')
   mclasses = (1,2,3,4,5,6,7,8,9)
   mcolors = [cmap(i) for i in np.linspace(0,1,10)]
   plt.figure(figsize=(12,8))
   for lab, col in zip(mclasses,mcolors):
      plt.scatter(Xred_pca[y == lab,0],Xred_pca[y == lab,1], label = lab, c = col)
   plt.xlabel('Componente 1')
   plt.ylabel('Componente 2')
   leg = plt.legend(loc= 'upper right', fancybox= True)
   plt.show()
   
def get_LDA(X_std,y):
   sklearn_lda = LDA(n_components=2)
   Xred_lda = sklearn_lda.fit_transform(X_std,y)
   cmap = plt.cm.get_cmap('Accent')
   mclasses=(1,2,3,4,5,6,7,8,9)
   mcolors = [cmap(i) for i in np.linspace(0,1,10)]
   plt.figure(figsize=(12, 8))

   for lab, col in zip(mclasses,mcolors):
      plt.scatter(Xred_lda[y == lab, 0],Xred_lda[y == lab, 1],label = lab,c = col)

   plt.xlabel('LDA/Fisher Direction 1')
   plt.ylabel('LDA/Fisher Direction 2')
   leg = plt.legend(loc='upper right', fancybox=True)
   plt.show()


def Dummy(X_std,y):
   dummy = DummyClassifier( strategy = 'stratified', random_state = None, constant = None)
   model = dummy.fit(X_std,y)
   prediction = model.predict(X_std)
   prediction_prob = model.predict_proba(X_std)

   plt.hist(prediction,bins=11)
   plt.show()
 
def get_performance(test_df,X_std,y):
   Xtest = test_df.ix[:,'x.1':'x.10'].values
   ytest = test_df.ix[:,'y'].values

   X_std_test = StandardScaler().fit_transform(Xtest)
   
   lda_model = LDA()
   lda_model.fit(X_std,y)
   
   qda_model = QDA()
   qda_model.fit(X_std,y)

   knn_model = KNeighborsClassifier(n_neighbors = 10)
   knn_model.fit(X_std,y)
   
   print "KNN SCORE"
   print knn_model.score(X_std_test,ytest)
   print "LDA SCORE"
   print lda_model.score(X_std_test,ytest)
   print "QDA SCORE"
   print qda_model.score(X_std_test,ytest)

   knn_scores_training = []
   knn_scores_test = []

   for i in range(1,12):
      knn_model = KNeighborsClassifier(n_neighbors = i)
      knn_model.fit(X_std,y)
      knn_scores_training.append(knn_model.score(X_std_test,ytest))
      knn_scores_test.append(knn_model.score(X_std,y))

   plt.plot(range(11),knn_scores_training,'r--')
   plt.plot(range(11),knn_scores_test,'b--')
   plt.axis([0,10,0.3,1.1])
   plt.show()



train_df , test_df = get_data(train_data_url,test_data_url)
X_std , y = scale_data(train_df, test_df)

#get_PCA(X_std)
#get_LDA(X_std,y)
#Dummy(X_std,y)

def get_PCA_performance(test_df,X_std,y):
   X_test = test_df.ix[:,'x.1':'x.10'].values
   y_test = test_df.ix[:,'y'].values
   X_std_test = StandardScaler().fit_transform(X_test)

   lda_scores_training = []
   lda_scores_test = []

   qda_scores_training = []
   qda_scores_test = []

   knn_scores_training = []
   knn_scores_test = []

   for d in range(1, 11):
      pca = PCA(n_components = d)
      Xred_pca_training = pca.fit_transform(X_std)
      Xred_pca_test = pca.transform(X_std_test)

      lda_model = LDA()
      lda_model.fit(Xred_pca_training , y)
       
      qda_model = QDA()
      qda_model.fit(Xred_pca_training , y)
       
      knn_model = KNeighborsClassifier(n_neighbors = 10)
      knn_model.fit(Xred_pca_training , y)

      lda_scores_training.append(1 - lda_model.score(Xred_pca_training ,y))
      lda_scores_test.append(1 - lda_model.score(Xred_pca_test,y_test))

      qda_scores_training.append(1 - qda_model.score(Xred_pca_training,y))
      qda_scores_test.append(1- qda_model.score(Xred_pca_test,y_test))

      knn_scores_training.append(1 - knn_model.score(Xred_pca_training,y))
      knn_scores_test.append(1 - knn_model.score(Xred_pca_test,y_test))

   plt.plot(range(10),lda_scores_training,'r--', label="Train data")
   plt.plot(range(10),lda_scores_test,'b--', label="Test data")
   plt.title("LDA vs PCA")
   plt.show()

   plt.plot(range(10),qda_scores_training,'r--', label="Train data")
   plt.plot(range(10),qda_scores_test,'b--', label="Test data" )
   plt.title("QDA vs PCA")
   plt.show()

   plt.plot(range(10),knn_scores_training,'r--', label="Train data")
   plt.plot(range(10),knn_scores_test,'b--', label="Test data")
   plt.title("KNN vs PCA")
   plt.show()

def get_LDA_performance(test_df,X_std,y):
   X_test = test_df.ix[:,'x.1':'x.10'].values
   X_std_test = StandardScaler().fit_transform(X_test)
   y_test = test_df.ix[:,'y'].values

   lda_scores_training = []
   lda_scores_test = []

   qda_scores_training = []
   qda_scores_test = []

   knn_scores_training = []
   knn_scores_test = []

   for d in range(1, 11):
      lda = LDA( n_components = d )
      Xred_lda_training = lda.fit_transform(X_std, y)
      Xred_lda_test = lda.transform(X_std_test)

      lda_model = LDA()
      lda_model.fit(Xred_lda_training , y)
       
      qda_model = QDA()
      qda_model.fit(Xred_lda_training , y)
       
      knn_model = KNeighborsClassifier(n_neighbors = 10)
      knn_model.fit(Xred_lda_training , y)

      lda_scores_training.append(1 - lda_model.score(Xred_lda_training ,y))
      lda_scores_test.append(1 - lda_model.score(Xred_lda_test,y_test))

      qda_scores_training.append(1 - qda_model.score(Xred_lda_training,y))
      qda_scores_test.append(1- qda_model.score(Xred_lda_test,y_test))

      knn_scores_training.append(1 - knn_model.score(Xred_lda_training,y))
      knn_scores_test.append(1 - knn_model.score(Xred_lda_test,y_test))

   plt.plot(range(10),lda_scores_training,'r--', label="Train data")
   plt.plot(range(10),lda_scores_test,'b--',label="Test data" )
   plt.title("LDA vs LDA")
   plt.xlabel('k')
   plt.ylabel('Score')
   plt.show()

   plt.plot(range(10),qda_scores_training,'r--', label="Train data")
   plt.plot(range(10),qda_scores_test,'b--', label="Test data")
   plt.title("QDA vs LDA")
   plt.show()

   plt.plot(range(10),knn_scores_training,'r--', label="Train data")
   plt.plot(range(10),knn_scores_test,'b--', label="Test data")
   plt.title("KNN vs LDA")
   plt.show()


get_PCA_performance(test_df,X_std,y)
get_LDA_performance(test_df,X_std,y)




