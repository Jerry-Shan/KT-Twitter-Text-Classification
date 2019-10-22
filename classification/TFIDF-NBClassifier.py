# coding:utf-8
# @Description : A text classifier using TFIDF to build the feature vectors and using Multinomial Naive Bayes as classifer
# @Author : Jinzhe Shan @ Unimelb
# @Data : 2019-10-13

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':

    tweetData = pd.read_csv('../dataset/train_tweets.txt', error_bad_lines=False)  # Skip the offending lines
    train_tweetContent = tweetData['content'].astype(str)
    train_labels = tweetData['class']

    tweetData = pd.read_csv('../dataset/dev_tweets.txt', error_bad_lines=False)  # Skip the offending lines
    dev_tweetContent = tweetData['content'].astype(str)
    dev_labels = tweetData['class']

    tweetData = pd.read_csv('../dataset/test_tweets.txt', error_bad_lines=False)  # Skip the offending lines
    test_tweetContent = tweetData['content'].astype(str)
    test_tweetID = tweetData['tweetID']

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'[a-z]{3,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=300)
    word_vectorizer.fit(train_tweetContent)
    print(word_vectorizer.get_feature_names())
    tfidf_words = word_vectorizer.get_feature_names()

    f = open('tfidf-words','w')
    f.write('tfidf words ' + '\n')
    for word in tfidf_words:
        f.write(word + '\n')
    f.close()
    print('save tfidf words successfully')

    train_features = word_vectorizer.transform(train_tweetContent)
    dev_features   = word_vectorizer.transform(dev_tweetContent)
    test_features  = word_vectorizer.transform(test_tweetContent)

    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

    # calculate Accuracy
    predicted_labels = clf.predict(dev_features)
    # print(dev_features,predicted_labels)

    accuracy = metrics.accuracy_score(dev_labels, predicted_labels)
    print('Accuracy：', '%.4f' % accuracy)

    # output for kaggle

    # test_lables = clf.predict(test_features)
    # f1 = open('best200_NB_Output_TFIDF.csv','a')
    # f1.write('tweet-id,'+'class'+'\n')
    # for i in range(len(test_lables)):
    #     f1.write(str(test_tweetID[i])+','+str(test_lables[i])+'\n')
    # f1.close()
    # print("the output is saved")
