# coding:utf-8
# @Description : A text classifier demo 4 using random forest
# @Author : Jinzhe Shan @ Unimelb
# @Data : 2019-10-10
# Accuracy = 0.6248

import pandas as pd
from sklearn import metrics, ensemble
import os
path = os.getcwd()

if __name__ == '__main__':
    # best 200
    train = pd.read_csv('../dataset/train-best200.csv').fillna(' ')
    test = pd.read_csv('../dataset/dev-best200.csv').fillna(' ')
    output = pd.read_csv('../dataset/test-best200.csv').fillna(' ')

    train_features,train_labels = train.iloc[:,:-1],train.iloc[:,-1]
    test_features, test_labels = test.iloc[:, :-1], test.iloc[:, -1]
    output_features = output.iloc[:, 1:-1]
    output_tweetID = output.iloc[:, 0].tolist()
    label_cols = ['NewYork','Georgia','California']
    features = output.iloc[0, 1:-1].tolist()
    print(features[:5])

    # random forest
    clf = ensemble.RandomForestClassifier(n_estimators=10).fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)

    # calculate Accuracy
    predicted_labels = clf.predict(test_features)
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    print('Accuracy：', '%.4f' % accuracy)

    print("importances : ")
    importances = clf.feature_importances_
    print(importances[:5])

    # output for kaggle
    output_labels = clf.predict(output_features)
    print(output_labels[:5])
    f1 = open('../output/best200_importances.csv','a')
    f1.write('features,'+'importance'+'\n')
    for i in range(len(importances)):
        f1.write(str(features[i])+','+str(importances[i])+'\n')
    f1.close()
    print("the output is saved")
