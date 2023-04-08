import pickle
import nltk
import pandas as pd
import string
import importlib.resources as resources
from pickle import load

personalities = {}
personalities[
    "ENTJ"
] = "Strategic, logical, efficient, outgoing, ambitious, independent Effective organizers of people and long-range planners"
personalities[
    "ENFJ"
] = "Caring, enthusiastic, idealistic, organized, diplomatic, responsible Skilled communicators who value connection with people"
personalities[
    "ESFJ"
] = "Friendly, outgoing, reliable, conscientious, organized, practical Seek to be helpful and please others, enjoy being active and productive"
personalities[
    "ESTJ"
] = "Efficient, outgoing, analytical, systematic, dependable, realistic Like to run the show and get things done in an orderly fashion"
personalities[
    "ENTP"
] = "Inventive, enthusiastic, strategic, enterprising, inquisitive, versatile Enjoy new ideas and challenges, value inspiration"
personalities[
    "ENFP"
] = "Enthusiastic, creative, spontaneous, optimistic, supportive, playful Value inspiration, enjoy starting new projects, see potential in others"
personalities[
    "ESFP"
] = "Playful, enthusiastic, friendly, spontaneous, tactful, flexible Have strong common sense, enjoy helping people in tangible ways"
personalities[
    "ESTP"
] = "Outgoing, realistic, action-oriented, curious, versatile, spontaneous Pragmatic problem solvers and skillful negotiators"
personalities[
    "INTP"
] = "Intellectual, logical, precise, reserved, flexible, imaginative Original thinkers who enjoy speculation and creative problem solving"
personalities[
    "INFP"
] = "Sensitive, creative, idealistic, perceptive, caring, loyal Value inner harmony and personal growth, focus on dreams and possibilities"
personalities[
    "ISFP"
] = "Gentle, sensitive, nurturing, helpful, flexible, realistic Seek to create a personal environment that is both beautiful and practical"
personalities[
    "ISTP"
] = "Action-oriented, logical, analytical, spontaneous, reserved, independent Enjoy adventure, skilled at understanding how mechanical things work"
personalities[
    "INTJ"
] = "Innovative, independent, strategic, logical, reserved, insightful Driven by their own original ideas to achieve improvements"
personalities[
    "INFJ"
] = "Idealistic, organized, insightful, dependable, compassionate, gentle Seek harmony and cooperation, enjoy intellectual stimulation"
personalities[
    "ISFJ"
] = "Warm, considerate, gentle, responsible, pragmatic, thorough Devoted caretakers who enjoy being helpful to others"
personalities[
    "ISTJ"
] = "Responsible, sincere, analytical, reserved, realistic, systematic Hardworking and trustworthy with sound practical judgment"


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

        results = ""

        if ie.lower() == "introvert":
            results += "I"
        if ie.lower() == "extrovert":
            results += "E"
        if Is.lower() == "intuition":
            results += "N"
        if Is.lower() == "sensing":
            results += "S"
        if tf.lower() == "thinking":
            results += "T"
        if tf.lower() == "feeling":
            results += "F"
        if jp.lower() == "judging":
            results += "J"
        if jp.lower() == "percieving":
            results += "P"

        return results

    def generate_key_value(self, results, input):
        if input in results:
            results[input] = results[input] + 1
        else:
            print("this should not happen")
            exit(-1)

    def generate_map_from_array(self, array, map):
        for item in array:
            map[item] = 0

    def describe_type(self, item):
        if item in personalities:
            return personalities[item]
        else:
            return ""

    def get_personality(self, input):
        results = {}
        a = [
            "ENFJ",
            "ENFP",
            "ENTJ",
            "ENTP",
            "ESFJ",
            "ESFP",
            "ESTJ",
            "ESTP",
            "INFJ",
            "INFP",
            "INTJ",
            "INTP",
            "ISFJ",
            "ISFP",
            "ISTJ",
            "ISTP",
        ]
        self.generate_map_from_array(a, results)
        data = self.find_data(input)
        self.generate_key_value(results, data)
        return results
