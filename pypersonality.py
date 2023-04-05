import pickle
import nltk
import pandas as pd
import string

def build_bag_of_words_features_filtered(words):
    useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in useless_words}

class PyPersonality:
    traits=['INTROVERT - EXTROVERT', 'INTUITION - SENSING', 'THINKING - FEELING', 'JUDGING - PERCIEIVING']

    def __init__(self,model_folder) -> None:
        self.model_folder=model_folder
        self.IntroExtro = self.load_classifier("models/IntroExtro.pickle")
        self.IntuitionSensing = self.load_classifier("models/IntuitionSensing.pickle")
        self.ThinkingFeeling = self.load_classifier("models/ThinkingFeeling.pickle")
        self.JudgingPercieiving = self.load_classifier("models/JudgingPercieiving.pickle")

    def load_classifier(self,filename):
        f = open(filename, 'rb')
        classifier = pickle.load(f)
        f.close()
        return classifier
    
    def MBTI(self,input):
        tokenize = build_bag_of_words_features_filtered(input)
        ie = self.IntroExtro.classify(tokenize)
        Is = self.IntuitionSensing.classify(tokenize)
        tf = self.ThinkingFeeling.classify(tokenize)
        jp = self.JudgingPercieiving.classify(tokenize)
        
        mbt = []

        mbt.append(ie.upper())
        mbt.append(Is.upper())
        mbt.append(tf.upper())
        mbt.append(jp.upper())
        
        return(mbt)

    def get_personality(self,input):
        print("general personality traits are")
        print(self.traits)
        print("in the given text i found the below")
        print(self.MBTI(input))
        return 



""" 
def MBTI(input):
    tokenize = build_bag_of_words_features_filtered(input)
    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)
    
    mbt = ''
    
    if(ie == 'introvert'):
        mbt+='I'
    if(ie == 'extrovert'):
        mbt+='E'
    if(Is == 'Intuition'):
        mbt+='N'
    if(Is == 'Sensing'):
        mbt+='S'
    if(tf == 'Thinking'):
        mbt+='T'
    if(tf == 'Feeling'):
        mbt+='F'
    if(jp == 'Judging'):
        mbt+='J'
    if(jp == 'Percieving'):
        mbt+='P'
    return(mbt) """
    
""" 

def tellmemyMBTI(input, name, traasits=[]):
    a = []
    trait1 = pd.DataFrame([0,0,0,0],['I','N','T','J'],['count'])
    trait2 = pd.DataFrame([0,0,0,0],['E','S','F','P'],['count'])
    for i in input:
        a += [MBTI(i)]
    for i in a:
        for j in ['I','N','T','J']:
            if(j in i):
                trait1.loc[j]+=1                
        for j in ['E','S','F','P']:
            if(j in i):
                trait2.loc[j]+=1 
    trait1 = trait1.T
    trait1 = trait1*100/len(input)
    trait2 = trait2.T
    trait2 = trait2*100/len(input)
    
    
    #Finding the personality
    YourTrait = ''
    for i,j in zip(trait1,trait2):
        temp = max(trait1[i][0],trait2[j][0])
        if(trait1[i][0]==temp):
            YourTrait += i  
        if(trait2[j][0]==temp):
            YourTrait += j
    traasits +=[YourTrait] 
    
    #Plotting
    
    labels = np.array(results.columns)

    intj = trait1.loc['count']
    ind = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, intj, width, color='royalblue')

    esfp = trait2.loc['count']
    rects2 = ax.bar(ind+width, esfp, width, color='seagreen')

    fig.set_size_inches(10, 7)
    
    

    ax.set_xlabel('Finding the MBTI Trait', size = 18)
    ax.set_ylabel('Trait Percent (%)', size = 18)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,105, step= 10))
    ax.set_title('Your Personality is '+YourTrait,size = 20)
    plt.grid(True)
    
    
    fig.savefig(name+'.png', dpi=200)
    
    plt.show()
    return(traasits)

trait=tellmemyMBTI("I really hate this way , I do not want you to help me", 'Divy') """


#https://www.kaggle.com/code/gokulkartha/mbti-personality-classifier/edit

p=PyPersonality("")
results=p.get_personality("Oh, thanks for inviting me. I appreciate it. I'm not sure if I'll be able to make it, though. I have some other things I need to take care of this weekend . I'm sure it will be a great time, but I'm just not feeling up for a big social gathering right now. Maybe we can plan something else another time?")

print(results)