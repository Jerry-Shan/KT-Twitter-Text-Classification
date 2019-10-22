# coding:utf-8
# @Description : A text classifier demo 1 using Multinomial Naive Bayes
# @Author : Jinzhe Shan @ Unimelb
# @Data : 2019-10-08

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

if __name__ == '__main__':

    # best 200
    train = pd.read_csv('../dataset/train-best200.csv').fillna(' ')
    valid = pd.read_csv('../dataset/dev-best200.csv').fillna(' ')
    test = pd.read_csv('../dataset/test-best200.csv').fillna(' ')

    train_features,train_labels = train.iloc[:,:-1],train.iloc[:,-1]
    valid_features, valid_labels = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_features = test.iloc[:, 1:-1]
    test_tweetID = test.iloc[:, 0].tolist()

    # 多项式贝叶斯分类器 Multinomial Naive Bayes
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
    predicted_labels = clf.predict(valid_features)
    accuracy = metrics.accuracy_score(valid_labels, predicted_labels)

    # calculate Accuracy
    predicted_labels = clf.predict(valid_features)
    accuracy = metrics.accuracy_score(valid_labels, predicted_labels)
    print('Accuracy：', '%.4f' % accuracy)

    # output for kaggle
    # output_labels = clf.predict(test_features)
    # print(output_labels[:5])
    # f1 = open('../output/best200_NB_Output.csv','a')
    # f1.write('tweet-id,'+'class'+'\n')
    # for i in range(len(output_labels)):
    #     f1.write(str(test_tweetID[i])+','+str(output_labels[i])+'\n')
    # f1.close()
    # print("the output is saved")
