import numpy as np

#Variables for MLP Classifier
hidden_layer_sizes = np.array([70,140,70])
alpha = 0.1
batch_size = 200
max_iter = 200


#my code for ann using sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

vec72_csv = pd.read_csv('vectors72_temp_withheaders.csv') #42334*74
vec72_temp = vec72_csv.sample(frac=1) #shuffling #42334*74

#making data for ANN
vec72 = vec72_temp.drop(vec72_temp.index[0:4]) #deleting first 4 rows #42330*74
ecg_id = vec72.loc[:,['ecg_id']] #ecg_id coloumn #42330*1
Annot = vec72.loc[:,['Annot']] #Annotation coloumn #42330*1
feat = vec72.drop(['ecg_id','Annot'],axis=1) #sub-matrix by removing ecg_id,Annotation coloumns #423440*72

#data for ANN
scaler = MinMaxScaler()
scaler.fit(feat)
X = scaler.transform(feat) #scaled features from R to (0,1) for each feature(coloumn) #42330*72
y = Annot.to_numpy() #42330*1
ID = ecg_id.to_numpy() #42330*1

#MLP Classifier
MyMLP = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, batch_size=batch_size, max_iter=max_iter, shuffle=False, verbose=False)

#10-fold cross validation
A = 0
n = 1
while (n < 2) :
    #indices of 10-fold cross validation
    N1 = (n-1)*4233
    N2 = n*4233
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
