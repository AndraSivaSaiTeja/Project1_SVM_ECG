def svmcg(t):
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import time
    pos_datafile0 = np.loadtxt('V4_ap.txt',dtype='f',delimiter=',')
    np.random.shuffle(pos_datafile0)
    neg_datafile0 = np.loadtxt('V4_an.txt',dtype='f',delimiter=',')
    np.random.shuffle(neg_datafile0)
    pos_datafile = np.round((pos_datafile0 * 1000))
    neg_datafile = np.round((neg_datafile0 * 1000))
    print(pos_datafile)
    print(neg_datafile)
    T = (2*t)+1
    M = np.zeros([T,T])
    C_range = np.logspace(-t,t,T)
    gamma_range = np.logspace(-t,t,T)
    begin = time.time()
    i = 0
    while (i < T):
        c1 = C_range[i]
        j = 0
        while (j < T):
            g1 = gamma_range[j]
            A=0
            n=1
            while (n < 11):
                N1 = (n-1)*50
                N2 = n*50
                test_pos_datafile = pos_datafile[N1:N2,:]
                train_pos_datafile_1 = pos_datafile[0:N1,:]
                train_pos_datafile_2 = pos_datafile[N2:500,:]
                train_pos_datafile = np.row_stack((train_pos_datafile_1,train_pos_datafile_2))
                test_neg_datafile = neg_datafile[N1:N2,:]
                train_neg_datafile_1 = neg_datafile[0:N1,:]
                train_neg_datafile_2 = neg_datafile[N2:500,:]
                train_neg_datafile = np.row_stack((train_neg_datafile_1,train_neg_datafile_2))
                train_datafile = np.row_stack((train_pos_datafile,train_neg_datafile))
                test_datafile = np.row_stack((test_pos_datafile,test_neg_datafile))
                X_train = train_datafile[:,[0,1,2]]
                y_train = train_datafile[:,3]
                X_test = test_datafile[:,[0,1,2]]
                y_test = test_datafile[:,3]
                svclassifier = SVC(C=c1,kernel='rbf',gamma=g1)
                svclassifier.fit(X_train,y_train)
                y_pred = svclassifier.predict(X_test)
                cm = confusion_matrix(y_test,y_pred)
                A = A+(((cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0]))*100) #Accuracy Sum
                n=n+1
            A0 = A/10
            M[i,j] = A0
            j=j+1
        load_time_0 = (((i+1)*100)/T)
        load_time_1 = int(load_time_0)
        print('Loading...',load_time_1,'%')
        i=i+1
    print(M)
    end = time.time()
    print(end-begin)
    return M
   #This function prints average accuracy(for 10 fold cross validation) matrix of rbf svm classifier for values of C and gamma from 1e-t to 1e-t
