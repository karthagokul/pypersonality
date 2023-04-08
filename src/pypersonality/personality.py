import pickle
import nltk
import pandas as pd
import string
import importlib.resources as resources
from pickle import load


def build_bag_of_words_features_filtered(words):
    useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    words = nltk.word_tokenize(words)
    return {word: 1 for word in words if not word in useless_words}


class PyPersonality:
    traits = ["INTROVERT - EXTROVERT", "INTUITION - SENSING", "THINKING - FEELING", "JUDGING - PERCIEIVING"]

    def __init__(self) -> None:
        self.IntroExtro = self.load_resource("models/IntroExtro.pickle")
        self.IntuitionSensing = self.load_resource("models/IntuitionSensing.pickle")
        self.ThinkingFeeling = self.load_resource("models/ThinkingFeeling.pickle")
        self.JudgingPercieiving = self.load_resource("models/JudgingPercieiving.pickle")

    def load_resource(self, file_name):
        clfier = None
        file_path = resources.files("pypersonality").joinpath(file_name)
        try:
            clfier = self.load_classifier(file_path)
        except FileNotFoundError:
            raise FileNotFoundError("")
        return clfier

    def load_classifier(self, filename):
        f = open(filename, "rb")
        classifier = pickle.load(f)
        f.close()
        return classifier

    def find_data(self, input):
        tokenize = build_bag_of_words_features_filtered(input)
        ie = self.IntroExtro.classify(tokenize)
        Is = self.IntuitionSensing.classify(tokenize)
        tf = self.ThinkingFeeling.classify(tokenize)
        jp = self.JudgingPercieiving.classify(tokenize)

        results = []

        results.append(ie.upper())
        results.append(Is.upper())
        results.append(tf.upper())
        results.append(jp.upper())

        return results

    def get_personality(self, input):
        results = {}
        results["info"] = self.traits
        results["results"] = self.find_data(input)
        return results
