import pandas as pd
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier
import pickle


class PyPersonalityTrainer:
    def __init__(self) -> None:
        pass

    def train(self, model_folder):
        data_set = pd.read_csv("input/mbti_personality_input.csv")
        data_set.tail()
        data_set.isnull().any()
        data_set.iloc[0, 1].split("|||")
        types = np.unique(np.array(data_set["type"]))
        total = data_set.groupby(["type"]).count() * 50
        all_posts = pd.DataFrame()
        for j in types:
            temp1 = data_set[data_set["type"] == j]["posts"]
            temp2 = []
            for i in temp1:
                temp2 += i.split("|||")
            temp3 = pd.Series(temp2)
            all_posts[j] = temp3
        all_posts.tail()
        useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

        def filter_results(words):
            words = nltk.word_tokenize(words)
            return {word: 1 for word in words if not word in useless_words}

        filter_results(all_posts["INTJ"].iloc[1])
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            features += [[(filter_results(i), j) for i in temp1]]
        split = []
        for i in range(16):
            split += [len(features[i]) * 0.8]
        split = np.array(split, dtype=int)
        train = []
        for i in range(16):
            train += features[i][: split[i]]
        sentiment_classifier = NaiveBayesClassifier.train(train)
        nltk.classify.util.accuracy(sentiment_classifier, train) * 100
        test = []
        for i in range(16):
            test += features[i][split[i] :]
        nltk.classify.util.accuracy(sentiment_classifier, test) * 100
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            if "I" in j:
                features += [[(filter_results(i), "introvert") for i in temp1]]
            if "E" in j:
                features += [[(filter_results(i), "extrovert") for i in temp1]]
        train = []
        for i in range(16):
            train += features[i][: split[i]]
        IntroExtro = NaiveBayesClassifier.train(train)
        nltk.classify.util.accuracy(IntroExtro, train) * 100
        test = []
        for i in range(16):
            test += features[i][split[i] :]
        nltk.classify.util.accuracy(IntroExtro, test) * 100
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            if "N" in j:
                features += [[(filter_results(i), "Intuition") for i in temp1]]
            if "E" in j:
                features += [[(filter_results(i), "Sensing") for i in temp1]]

        train = []
        for i in range(16):
            train += features[i][: split[i]]
        IntuitionSensing = NaiveBayesClassifier.train(train)
        nltk.classify.util.accuracy(IntuitionSensing, train) * 100
        test = []
        for i in range(16):
            test += features[i][split[i] :]

        nltk.classify.util.accuracy(IntuitionSensing, test) * 100

        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            if "T" in j:
                features += [[(filter_results(i), "Thinking") for i in temp1]]
            if "F" in j:
                features += [[(filter_results(i), "Feeling") for i in temp1]]

        train = []
        for i in range(16):
            train += features[i][: split[i]]

        ThinkingFeeling = NaiveBayesClassifier.train(train)

        nltk.classify.util.accuracy(ThinkingFeeling, train) * 100
        test = []
        for i in range(16):
            test += features[i][split[i] :]
        nltk.classify.util.accuracy(ThinkingFeeling, test) * 100

        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            if "J" in j:
                features += [[(filter_results(i), "Judging") for i in temp1]]
            if "P" in j:
                features += [[(filter_results(i), "Percieving") for i in temp1]]
        train = []
        for i in range(16):
            train += features[i][: split[i]]

        JudgingPercieiving = NaiveBayesClassifier.train(train)
        nltk.classify.util.accuracy(JudgingPercieiving, train) * 100

        test = []
        for i in range(16):
            test += features[i][split[i] :]

        nltk.classify.util.accuracy(JudgingPercieiving, test) * 100

        # Lets save the trained models
        self.save_classifier(IntroExtro, "models/IntroExtro.pickle")
        self.save_classifier(IntuitionSensing, "models/IntuitionSensing.pickle")
        self.save_classifier(ThinkingFeeling, "models/ThinkingFeeling.pickle")
        self.save_classifier(JudgingPercieiving, "models/JudgingPercieiving.pickle")

    def save_classifier(self, classifier, file_name):
        f = open(file_name, "wb")
        pickle.dump(classifier, f, -1)
        f.close()


p = PyPersonalityTrainer()
p.train("")
