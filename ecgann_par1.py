import numpy as np

#Variables for MLP Classifier
hidden_layer_sizes = np.array([200,400,800,400,200])
alpha = 0.1
batch_size = 200
max_iter = 200


#my code for ann using sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

vec108_csv = pd.read_csv('vectors108_temp_withheaders.csv') #38285*110
vec108_temp = vec108_csv.sample(frac=1) #shuffling #38285*110

#making data for ANN
vec108 = vec108_temp.drop(vec108_temp.index[0:5]) #deleting first 5 rows #38280*110
ecg_id = vec108.loc[:,['ecg_id']] #ecg_id coloumn #38280*1
Annot = vec108.loc[:,['Annot']] #Annotation coloumn #38280*1
feat = vec108.drop(['ecg_id','Annot'],axis=1) #sub-matrix by removing ecg_id,Annotation coloumns #38280*108

#data for ANN
scaler = MinMaxScaler()
scaler.fit(feat)
X = scaler.transform(feat) #scaled features from R to (0,1) for each feature(coloumn) #38280*108
y = Annot.to_numpy() #38280*1
ID = ecg_id.to_numpy() #38280*1

#MLP Classifier
MyMLP = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, batch_size=batch_size, max_iter=max_iter, shuffle=False, verbose=False)

#10-fold cross validation
A = 0
n = 1
while (n < 2) :
    #indices of 10-fold cross validation
    N1 = (n-1)*3828
    N2 = n*3828
    #train test split
    X_train = np.delete(X,slice(N1,N2),0)
    y_train = np.delete(y,slice(N1,N2),0)
    ID_train = np.delete(ID,slice(N1,N2),0)
    X_test = X[N1:N2]
    y_test = y[N1:N2,:]
    ID_test = ID[N1:N2,:]
    #ANN train
    clf = MyMLP.fit(X_train, y_train.ravel())
    #ANN test
    y_pred = clf.predict(X_test)
    #Confusion matrix,Accuracy
    cm = confusion_matrix(y_test,y_pred)
    Acc = accuracy_score(y_test, y_pred)*100
    print('For n = ',n, ' Accuracy is ',Acc)
    A = A + Acc
    n=n+1
#print('Avg. Accuracy    : ',A/10)
