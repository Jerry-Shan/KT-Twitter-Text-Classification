
# @Description : A text classifier demo 0 using Zero-R as Baseline
# @Author : Jinzhe Shan @ Unimel
# @Data : 2019-10-07
# Baseline Accuracy： 0.6441

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

if __name__ == '__main__':

    # best 200
    train = pd.read_csv('../dataset/train-best200.csv').fillna(' ')
    test = pd.read_csv('../dataset/dev-best200.csv').fillna(' ')

    train_features,train_labels = train.iloc[:,2:-1],train.iloc[:,-1]
    test_features, test_labels = test.iloc[:, 2:-1], test.iloc[:, -1]

    # calculate Accuracy
    predicted_labels = ['NewYork'] * len(test_features)
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    print('Baseline Accuracy：', '%.4f' % accuracy)

