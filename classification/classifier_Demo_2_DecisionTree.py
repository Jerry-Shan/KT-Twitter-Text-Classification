# coding:utf-8
# @Description : A text classifier demo 2 using Decision Trees
# @Author : Jinzhe Shan @ Unimelb
# @Data : 2019-10-09

import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

if __name__ == '__main__':

    # best 200
    train = pd.read_csv('../dataset/train-best200.csv').fillna(' ')
    test = pd.read_csv('../dataset/dev-best200.csv').fillna(' ')
    output = pd.read_csv('../dataset/test-best200.csv').fillna(' ')

    train_features,train_labels = train.iloc[:,:-1],train.iloc[:,-1]
    test_features, test_labels = test.iloc[:, :-1], test.iloc[:, -1]
    output_features = output.iloc[:, :-1]
    output_tweetID = output.iloc[:, 0].tolist()
    print(output_tweetID[:5])
    label_cols = ['NewYork','Georgia','California']

    # Decision Trees
    classifier = DecisionTreeClassifier()
    classifier.fit(train_features, train_labels)

    # calculate Accuracy
    predicted_labels = classifier.predict(test_features)
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    print('Accuracy：', '%.4f' % accuracy)

    # output for kaggle

    # output_labels = classifier.predict(output_features)
    # print(output_labels[:5])
    # f1 = open('../output/best200_Output.csv','a')
    # f1.write('tweet-id,'+'class'+'\n')
    # for i in range(len(output_labels)):
    #     f1.write(str(output_tweetID[i])+','+str(output_labels[i])+'\n')
    # f1.close()
    # print("the output is saved")
