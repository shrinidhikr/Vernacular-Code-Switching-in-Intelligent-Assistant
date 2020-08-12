
# coding: utf-8

# In[40]:





# In[11]:


# Importing the libraries
import pycrfsuite
import numpy as np
import re
import pandas as pd
import enchant
from googletrans import Translator 
import requests
from bs4 import BeautifulSoup
import joblib
import itertools
import covid19
import nltk
from indictrans import Transliterator
from litcm import LIT
import ner_module
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
nltk.download('stopwords')
# nltk.download('words')
# from nltk.corpus import words

tagger = pycrfsuite.Tagger()
tagger.open('pos_crf')

#stopwords = pd.read_csv('hindi_and_english_stop_words.csv')['stop_words'].tolist()

lit = LIT(labels=['hin', 'eng','kan'], transliteration=False)
lit_det = LIT(labels=['eng','kan'], transliteration=False)
t = Translator()

def get_lang_cs(query,lit):
    words = query.split(" ")
    kan = 0
    eng = 0
    hin = 0
    for word in words:
        word = word.lower()
        word = word.strip()
        #print (lit.identify(word))
        if re.match("[a-z]+",word):
         if lit.identify(word).split("\\")[1][0:3].strip() == "Eng":
             eng+=1
         if lit.identify(word).split("\\")[1][0:3].strip() == "Kan":
             kan+=1
         if lit.identify(word).split("\\")[1][0:3].strip() == "Hin":
             hin+=1
   
    if (eng + kan) >= (eng + hin):
        
        return "en_ka"
    else: 
        return "en_hi"
def translate_engkan(sentence,t):
    
    
    
    return t.translate(sentence,dest = "en",src = "kn").text
def get_lang(q,lit_det):
    #print lit_det.identify(q)
    if re.match("[a-z]+",q):
     return lit_det.identify(q).split("\\")[1][0:3].strip()

def get_pos_tags(q):
    words = nltk.word_tokenize(q)
    treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
    #print treebankTagger.tag(words)
    return treebankTagger.tag(words)
def keyword_extract(q):
    #print q
    lis = []
    for word in q.split(" "):
        if get_lang(word,lit_det) == "Kan":
            tra = translate_engkan(word,t)
           
            lis.append(tra)
        else:
            lis.append(word)
    kan_to_eng = " ".join(lis)
    tags = get_pos_tags(kan_to_eng)
    valtags_1 = " ".join([word[0] for word in tags if word[1]=="WRB" or word[1] == "WP"])
    valtags = " ".join([word[0]  for word in tags if word[1] == "NN" or word[1] == "NNP" or word[1] == "VB" or word[1] == "JJ"])
    return valtags_1+" "+valtags

def asciiPercentage(s):
	count = 0
	for char in s:
		if ord(char) < 128:
			count += 1
	return count/len(s)

def vowelPercentage(s):
	vowels = "aeiou"
	count = 0.
	for char in s:
		if char in vowels:
			count += 1
	return count/len(s)

def word2features(sent, i):

	# feature vector
	# word, pos, lang

    word = sent[i][0]
    wordClean = ''.join([ch for ch in word if ch in 'asdfghjklqwertyuiopzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM']).lower()
    normalizedWord = wordClean.lower()
    
    anyCap = any(char.isupper() for char in word)
    hasSpecial = any(ord(char) > 32 and ord(char) < 65 for char in word)
    lang = sent[i][1]

    
    features = {'word' : word, 'wordClean' : wordClean, 'normalizedWord' : normalizedWord,                 'lang' : lang,
                'isTitle' : word.istitle(), 'wordLength' : len(word), \
                'anyCap' : anyCap, 'allCap' : word.isupper(),
                'hasSpecial' : hasSpecial, 'asciiPer' : asciiPercentage(word)}
    
    features['suffix3'] = word[-3:]
    features['prefix3'] = word[:3]
    features['suffix2'] = word[-2:]
    features['prefix2'] = word[:2]
    features['suffix1'] = word[-1:]
    features['prefix1'] = word[:1]  

    return features

def sent2features(sent):
	features = []

	for i in range(len(sent)):
		features.append(word2features(sent, i))

	return features

def sent2labels(sent):
	allLabels = []

	for i in sent:
		allLabels.append(i[2])

	return allLabels

def sent2tokens(sent):

	allTokens = []

	for i in sent:
		allTokens.append(i[0])

	return allTokens

def translate(sentence):
    translator = Translator()
    #print("trans---------")
    #print(translator.translate(sentence,dest = "en",src = "hi").text)
    return translator.translate(sentence,dest = "en",src = "hi").text

def preprocess_pipeline(sent):
    stop_list = []#pd.read_csv("hindi_and_english_stop_words.csv")["stop_words"].tolist()
    d = enchant.Dict("en_US")
    vec = []
    for word in sent.split(" "):
        for hword in stop_list:
            if word == hword:
                sent = sent.replace(word,"")
    
    for word in sent.split(" "):
        word = word.strip()
        if len(word) >= 2 and re.match("^[A-Za-z]+$",word):
            
            if d.check(word):
                vec.append([word,"en"])
            else:
                
                vec.append([word,"hi"])
        else:
            pass

    fea_vec = sent2features(vec)
    def check_lang(word,translator):
        return translator.detect(word).lang
    transl = []
    translator = Translator()

    for word in sent.split(" "):
        word = word.strip()
        if len(word) >= 2 and re.match("^[A-Za-z]+$",word):
            if check_lang(word,translator) == "en":
                transl.append([word,"en"])
            else:
                if word == "aadhaar":
                    transl.append([word,"hi"])
                else:
                    transl.append([translate(word),"hi"])
        else:
            pass
    tags = tagger.tag(fea_vec)

    #printtags
    for i in range(len(tags)):
        fea_vec[i]["postag"] = tags[i]
        #print fea_vec[i]
    
    word_tag_dict = {}
    test_word_tag_dict = {}
    #if fea_vec[index]["word"] not in ["book","note","remind","set","baje","ghante","oclock"] and (tags[index] == "NOUN" or tags[index] == "ADJ"):
            #activity.append(fea_vec[index]["word"])
    sent_dict = {}
    sent_dict["PROP_WH"] = ""
    sent_dict["NOUN"] = []
    sent_dict["VERB"] = ""
    sent_dict["ADJ"] = []
    sent_dict["PREP"] = ""
    sent_dict["PROPN"] = ""
    #prop_wh|verb adj noun*
    #print transl
    for i in range(0,len(transl)):
            if tags[i] == "PRON_WH":
                sent_dict["PROP_WH"] = transl[i][0]
                break
    for i in range(0,len(transl)):
            if tags[i] == "PROPN":
                sent_dict["PROPN"] = transl[i][0]
                break
    
    for i in range(0,len(transl)):
            if tags[i] == "NOUN":
                sent_dict["NOUN"].append(transl[i][0])
                
    for i in range(0,len(transl)):
        if tags[i] == "VERB":
            sent_dict["VERB"] = transl[i][0]
            break
    for i in range(0,len(transl)):
        if tags[i] == "PREP":
            sent_dict["PREP"] = transl[i][0]
            break
    for i in range(0,len(transl)):
        if tags[i] == "ADJ":
            sent_dict["ADJ"].append(transl[i][0])
    
    
    
    final_sent =  sent_dict["PROP_WH"]+" "+sent_dict["VERB"] + " "+" ".join(sent_dict["ADJ"]) + " "+sent_dict["PREP"]+" "+ sent_dict["PROPN"]+" "+" ".join(sent_dict["NOUN"])
    
    
           
    return final_sent
def load_cv_transform():
    df = pd.read_csv("dataset.csv",encoding = "latin-1")
    df = df.drop('High level class',axis=1)
    df['category_id'] = df['Lower level class'].factorize()[0]
    category_id_df = df[['Lower level class', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Lower level class']].values)
    pdf = df
    df = df.drop(['Lower level class'],axis=1)

# Cleaning the texts
    corpus = []
    for i in range(0, df.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', df['Code switched query'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values
    return cv,id_to_category
def kannada_load_cv_transform():
    df = pd.read_csv("Kannada dataset - Full_dataset.csv",encoding = "latin-1")
    df = df.drop('High level class',axis=1)
    df['category_id'] = df['Lower level class'].factorize()[0]
    category_id_df = df[['Lower level class', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Lower level class']].values)
    pdf = df
    df = df.drop(['Lower level class'],axis=1)

# Cleaning the texts
    corpus = []
    for i in range(0, df.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', df['Code switched query'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values
    return cv,id_to_category



def to_feature(query):
    cv,id_to_category = load_cv_transform()
    review = re.sub('[^a-zA-Z]', ' ', query)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = [review]
    return cv.transform(review),id_to_category
def kannada_to_feature(query):
    cv,id_to_category = kannada_load_cv_transform()
    review = re.sub('[^a-zA-Z]', ' ', query)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = [review]
    return cv.transform(review),id_to_category
def kannada_load_and_query_classifier(query):
    kclassifier = joblib.load('kanglishintent')
    fv,id_to_category = kannada_to_feature(query)
    print ("entered kannada classifier!")
    return id_to_category[kclassifier.predict(fv)[0]]

def load_and_query_classifier(query):
    classifier = joblib.load('hinglishintent')
    fv,id_to_category = to_feature(query)
    return id_to_category[classifier.predict(fv)[0]]
def getSearchResult(user_input):
    google_search = requests.get('https://www.google.com/search?q='+user_input)
    soup = BeautifulSoup(google_search.text,'html.parser')
    #print(soup.prettify())

    search_results = soup.select('div.ZINbbc.xpd.O9g5cc.uUPGi')
    #print(search_results)
    result_list = []
    for link in search_results[:5]:
        l = link.find_all("div",class_="kCrYT")
        if(len(l)!=0):
           #print(l)
            description = None
            title = l[0].find("div",class_="BNeawe vvjwJb AP7Wnd")
            link = l[0].find("a")
            if(len(l)>1):
                description = l[1].find("div",class_="BNeawe s3v9rd AP7Wnd")
            if(title!=None and description!=None and link!=None):
                result_list.append(title.text+"\n"+"https://www.google.com"+link.get('href'))
    
    response_string = "\n".join(result_list)
    #print(response_string)

    return response_string
def google_search(query,word_map,intent):
    
    google_string = word_map
    return "Extracted keywords required to query or perform task:  "+google_string+"\n"+"Results: "+"\n"+getSearchResult(google_string)
def transliterate(query,hin_trans,kan_trans):
     query = hin_trans.transform(query)
     query = kan_trans.transform(query)
     return query

def perform_action(query):
    query = str(query).lower()
    display_map = {}
	
    kan_trans = Transliterator(source = "kan",target = "eng",build_lookup=True)
    hin_trans = Transliterator(source = "hin",target = "eng",build_lookup=True)
    query = transliterate(query,hin_trans,kan_trans)
    if get_lang_cs(query,lit) == "en_hi":
        intent = load_and_query_classifier(query)
    if get_lang_cs(query,lit) == "en_ka":

        intent = kannada_load_and_query_classifier(query)


    #print(intent)
    #print ("INTENT identified:"+" "+intent)
    ner_type = ["HOTEL","RESTAURANT","TRAVEL_BOOKING","REMINDER"]
    
#    flag = 1
    
    for kind in ner_type:
        if kind == intent:
            return ner_module.response(query,intent)
    if get_lang_cs(query,lit) == "en_hi":
        #print "This is a query of type:en_hi"
        word_map = preprocess_pipeline(query)
    elif get_lang_cs(query,lit) == "en_ka":
        #print "This is a query of type:en_ka"
        word_map = keyword_extract(query)
    if intent in ["SYMPTOMS","TREATMENT","PREVENTION"]:
     if query.find("corona") >= 0 or query.find("covid") >= 0 or query.find("crna") >=0:
      if  intent == "SYMPTOMS":
       return covid19.symptoms
      elif intent == "TREATMENT":
       return covid19.treatment
      elif intent == "PREVENTION":
       return covid19.prevention
    #print(word_map)
    display_map["code_switch_type"] = get_lang_cs(query,lit)
    display_map["transliterated"] = query
    display_map["intent"] = intent
	
    return "Response:"+"\n"+str(display_map)+"\n"+google_search(query,word_map,intent)



# In[14]:




