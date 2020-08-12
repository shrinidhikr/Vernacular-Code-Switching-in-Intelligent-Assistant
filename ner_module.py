import pandas as pd
import dateparser
import datetime
from email_scraper import scrape_emails
import pandas as pd
import re


def get_hotel_name(query):
        hotel_map = set()
        hotel_list = pd.read_csv("hotel_list.csv")
        hotel_list.columns = ["index","hotel_name"]
        hotel_list = hotel_list["hotel_name"].tolist()
        
        hotel_map = list(hotel_list)
        response = set()
        
        for w in hotel_map:
                w = str(w)
                w = w.strip()
                if query.find(w) >= 0 and len(w) > 3:
                    #print w
                    response.add(w.strip())
        if  len(response) == 0:
            return "None"
        return list(response)
        
def get_cuisine(query):
        cuisine_map = set()
        cuisine_list = pd.read_csv("cuisine_list.csv")
        cuisine_list.columns = ["index","cuisine_name"]
        cuisine_list = cuisine_list["cuisine_name"].tolist()
        
        cuisine_map = list(cuisine_list)
        #print cuisine_map[0:100]
        response = set()
        
        for w in cuisine_map:
                w = str(w)
                w = w.strip()
                
                if query.find(w) >= 0 and len(w)>=3: 
                    #print w
                    response.add(w.strip())
        if  len(response) == 0:
            return "None"
        return list(response)
        
def get_restaraunt_name(query):
        restaurant_map = set()
        restaurant_list = pd.read_csv("restaurant_list.csv")
        restaurant_list.columns = ["index","name_p"]
        restaurant_list = restaurant_list["name_p"].tolist()
        
        restaurant_map = list(restaurant_list)
        response = set()
        
        for w in restaurant_map:
                w = str(w)
                w = w.strip()
                if query.find(w) >=0 and len(w)>3:
                    
                    response.add(w.strip())
        if  len(response) == 0:
            return "None"
        return list(response)
        
def get_vehicle_type(query):
    vec_list = pd.read_csv("vehicles.csv")
    vec_list.columns = ["vehicle_name"]
    vec_list = vec_list["vehicle_name"].tolist()
    response = set()
    for w in vec_list:
                w = str(w)
                w = w.strip()
                if query.find(w) >=0 and len(w)>=3:
                    
                    response.add(w.strip())
    if  len(response) == 0:
        return "None"
    return list(response)
    
def get_location(query):
        loc_map = set()
        loc_list = pd.read_csv("location_list.csv")
        loc_list.columns = ["index","location_name"]
        loc_list = loc_list["location_name"].tolist()
        
        loc_map = list(loc_list)
        loc1_list = []
        
        for word in query.split(" "):
            for w in loc_map:
                if word == w:
                    loc1_list.append(w)
        return loc1_list

def normalize_date(query):
    num_df = pd.read_csv("month_list.csv")
    numap = {}
    for index,row in num_df.iterrows():
        numap[row["month"]] = str(row["normalized_month"])
    
    resp = query
    for word in query.split(" "):
        if word in numap.keys():
            
            resp = resp.replace(word,numap[word])
    return resp

def get_date(query):
        #if aaj, get todays date
        #if kal, get tommorows date
        #if neither, date parse
		query = normalize_date(query)
		if query.find("aaj")>=0 or query.find("today")>=0  or query.find("ivathu")>=0:
			return datetime.datetime.today()
		if query.find("kal")>=0 or query.find("tomorrow")>=0 or query.find("nale")>=0:
			return datetime.datetime.today()+ datetime.timedelta(days=1)
		from dateutil.parser import parse
		month_lis = pd.read_csv("month_list.csv")["normalized_month"].tolist()
		for month in month_lis:
			if re.match(".*"+month+".*",query):
				if parse(query,fuzzy=True).year is not datetime.datetime.today().year:
					dateobj = parse(query,fuzzy=True)
					dateobj = dateobj.replace(year = 2020)
					return dateobj
				else:
					return parse(query,fuzzy = True)
            	
		return "None"

def to_numeric(query):
    num_df = pd.read_csv("word2num.csv")
    numap = {}
    for index,row in num_df.iterrows():
        numap[row["english_number"]] = str(row["english_numeral"])
    
    resp = query
    for word in query.split(" "):
        if word in numap.keys():
            
            resp = resp.replace(word,numap[word]) 
    return resp

def get_time(query):
        query = to_numeric(query)
        if query.find("shaam") >= 0 or query.find("evening")>=0 or query.find("sainkala")>=0:
            return "18:00"
        elif query.find("night") >= 0 or query.find("raathri")>=0 or query.find("raath")>=0:
            return "21:00"
        elif query.find("morning") >= 0 or query.find("subaha")>=0 or query.find("belagatha")>=0:
            return "9:00"
        time_abs_re = ".*([0-9][0-9]:00).*|.*([0-9]:00).*"
        time_pm_re =".*([0-9][0-9]).(pm|PM).*|.*([0-9]).(pm|PM).*"
        time_am_re = ".*([0-9][0-9]).(am|AM).*|.*([0-9]).(am|AM).*"
        time_re = ".*([0-9][0-9]).(baje|(o clock)|o'clock|oclock|gante|ghante).*|.*([0-9]).(baje|o'clock|(o clock)|oclock|gante|ghante).*"
        n = ""
        if re.match(time_abs_re,query):
             return re.findall(time_abs_re,query)[0]
        if re.match(time_am_re,query):
            #print re.findall(time_am_re,query)
            for i in re.findall(time_am_re,query)[0]:
                if i.isdigit():
                    n = i
            return n+":00"
        elif re.match(time_pm_re,query):
            #print re.findall(time_pm_re,query)
            for i in re.findall(time_pm_re,query)[0]:
                if i.isdigit():
                    n = i
            return str(int(n)+12) +":00"
        elif re.match(time_re,query):
            
            #print re.findall(time_re,query)
            if re.findall(time_re,query)[0][0] ==  "":
                
                return str(int(re.findall(time_re,query)[0][3])+12)+":00"
            else:
                return str(int(re.findall(time_re,query)[0][0])+12)+":00"

        else:
            return "None"

def get_emailid(query):
        return list(scrape_emails(query))

def get_no_of_reservations(query):
        query = to_numeric(query)
    
        room_re = ".*([0-9]+).*(room|rooms|table|tables|gaadi|talika|taliko|kaksh|kaksho|ticket|tickets|log|persons|people|logon|janarige|jana|janaru).*"
        if re.match(room_re,query):
            return re.findall("[0-9]+",query)[0]
        else:
            return "None"

def get_no_of_people(query):
    query = to_numeric(query)
    
    people_re = ".*([0-9]+).*(log|persons|people|logon|janarige|jana|janaru).*"
    if re.match(people_re,query):
        return re.findall("[0-9]+",query)[0]
    else:
        return "None"

def get_phno(query):
    if re.match(".*([0-9]{10}).*",query):
        return re.findall('.*([0-9]{10}).*',query)[0]
    else:
        return "None"


import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report
tagger = pycrfsuite.Tagger()
tagger.open('pos_crf')

def asciiPercentage(s):
	count = 0.
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
    allCap = all(char.isupper() for char in word)
    hasSpecial = any(ord(char) > 32 and ord(char) < 65 for char in word)
    lang = sent[i][1]
    
    hashTag = word[0] == '#'
    mention = word[0] == '@'
    
    
    features = {'word' : word, 'wordClean' : wordClean, 'normalizedWord' : normalizedWord, 'lang' : lang,
                'isTitle' : word.istitle(), 'wordLength' : len(word), \
                'anyCap' : anyCap, 'allCap' : word.isupper(),
                'hasSpecial' : hasSpecial, 'asciiPer' : asciiPercentage(word)}
    
    
#     features['suffix5'] = word[-5:]
#     features['prefix5'] = word[:5]
#     features['suffix4'] = word[-4:]
#     features['prefix4'] = word[:4]
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


import enchant
import re
import pandas as pd
import googletrans
from googletrans import Translator 

def translate(sentence):
    translator = Translator()

    return translator.translate(sentence,dest = "en",src = "hi").text

def get_activity(sent):
    stop_list = []#pd.read_csv("hindi_and_english_stop_words.csv")["stop_words"].tolist()
    d = enchant.Dict("en_US")
    vec = []
    sent = to_numeric(sent)
    sent = re.sub("[0-9]","",sent)
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
    transl = []
    translator = Translator()

    
    tags = tagger.tag(fea_vec)
    #print tags
    activity = []
    sent = sent.lower()
    sent = re.sub("[^a-z ]","",sent)
    sent = sent.split(" ")
    #print sent
    months = pd.read_csv("month_list.csv")["normalized_month"].tolist()
    time_words = pd.read_csv("activity_stopwords.csv")["word"].tolist()
    
    for index in range(len(tags)):
        if fea_vec[index]["word"] not in ["book","note","remind","set","baje","ghante","oclock"] and (tags[index] == "NOUN" or tags[index] == "ADJ") and fea_vec[index]["word"] not in months and len(fea_vec[index]["word"]) > 2 and fea_vec[index]["word"] not in time_words:
            #print activity
            activity.append(fea_vec[index]["word"])
    activity = " ".join(activity)
    return activity

    
    
   
# Importing the libraries


#print preprocess_pipeline("malaria ka symptoms kya hai")

#print preprocess_pipeline("goverment card kaise update karu")

#print preprocess_pipeline("nearest darji kidhar hai")

#print preprocess_pipeline("Swiggy Delivery ka time kitana lagega")

#Set an alarm 6 baje ke liye
#Remind me kal doctor ka appointment hai
#Remind me chaar baje movie jaana hai
#Note kal dentist ka appointment hai
    
def perform_entity_extraction(query,entity_list):
        response = {}
    
        for entity in entity_list:
            if entity == "hotel_name":
                response[entity] = get_hotel_name(query)
            if entity == "restaurant_name":
                response[entity] = get_restaraunt_name(query)
            if entity == "from_location":
                response[entity] = get_location(query)[0]
            if entity == "to_location":
                response[entity] = get_location(query)[1]
            if entity == "date":
                response[entity] = get_date(query)
            if entity == "no_of_people":
                response[entity] = get_no_of_people(query)
            if entity == "no_of_reservations":
                response[entity] = get_no_of_reservations(query)
            if entity == "time":
                response[entity] = get_time(query)
            if entity == "activity":
                response[entity] = get_activity(query)
            if entity == "email_id":
                response[entity] = get_emailid(query)
            if entity == "ph_no":
                response[entity] = get_phno(query)
            if entity == "vehicle_type":
                response[entity] = get_vehicle_type(query)
            if entity == "cuisine":
                response[entity] = get_cuisine(query)
        return response

         
# Travel Booking : Ask for details if missed, From To Date no_of_people
# Hotel Booking:
# https://bus.makemytrip.com/bus/search/Bangalore/Davanagere/10-08-2020  
# https://railways.makemytrip.com/listing/?classCode=&date=20200813&destCity=Chennai&srcCity=Bangalore
# https://www.makemytrip.com/flight/search?itinerary=FROM_LOC-TO_LOC-DATE-FROM_LOC-TO_LOC-DATE&tripType=R&paxType=A-NO_OF_TICKETS_C-0_I-0&intl=false&cabinClass=E

def to_readable(date):
    dd,mm = get_dd_mm(date)
    return dd+"/"+mm+"/"+str(date.year)

def entity_is_null(response):
    null_keys = []
    for k in response.keys():
        if response[k] == "None":
            null_keys.append(k)
    if len(null_keys) == 0:
        return 0
    else:
        return ",".join(null_keys)

def get_dd_mm(date):
    dd = ''
    mm = ''
    if(1<=date.day and date.day<=9):
        dd = str('0') + str(date.day)
    else:
        dd = str(date.day)
    if(1<=date.month and date.month<=9):
        mm = str('0') + str(date.month)
    else:
        mm = str(date.month)
    return dd, mm

def response(query,intent_name):
    query = str(query).lower()
    if intent_name == "TRAVEL_BOOKING":
        
        response = perform_entity_extraction(query,["from_location","to_location","vehicle_type","date","no_of_reservations","time"])
        print(response)
        if entity_is_null(response) != 0:
            return "Please type the query with the following datapoints referenced:"+ entity_is_null(response)
        else:
            if(response['vehicle_type'][0].lower() == 'bus'):
                dd, mm = get_dd_mm(response['date'])
                return "Booking Results: Please follow this link to book your ticket:"+"https://bus.makemytrip.com/bus/search/"+response['from_location'].capitalize()+"/"+response['to_location'].capitalize()+"/"+str(dd)+"-"+str(mm)+"-"+str(response['date'].year)
            if(response['vehicle_type'][0].lower() == 'train'):
                dd, mm = get_dd_mm(response['date'])
                return "Booking Results: Please follow this link to book your ticket:"+"https://railways.makemytrip.com/listing/?classCode=&date="+str(response['date'].year)+str(mm)+str(dd)+"&destCity="+response['to_location'].capitalize()+"&srcCity="+response['from_location'].capitalize()
            if(response['vehicle_type'][0].lower() == 'flight'):
                return "Booking Results: Please follow this link to book your ticket:"+"https://www.makemytrip.com/"+str(response["vehicle_type"][0])+"/search?itinerary="+response["from_location"][0:3].upper()+"-"+response["to_location"][0:3].upper()+"-"+to_readable(response["date"])+"&tripType=O&paxType=A-"+response["no_of_reservations"]+"_C-0_I-0&intl=false&cabinClass=E"
            if(response['vehicle_type'][0].lower() not in ['bus','train','flight']):
                return "Booked "+response["vehicle_type"][0]+" for "+str(response["no_of_reservations"])+" from "+response["from_location"]+" to "+response["to_location"]+" on "+to_readable(response["date"])+ " at "+response["time"]
    
    elif intent_name == "HOTEL":
        response = perform_entity_extraction(query,["date","no_of_reservations","hotel_name"])
        
        if entity_is_null(response) != 0:
            return "Please type the query with the following datapoints referenced:"+ entity_is_null(response)
        else:
            return "Booking Confirmed for "+response["hotel_name"][1]+" "+"for date "+to_readable(response["date"])+" "+"with no of reservations "+str(response["no_of_reservations"])
    elif intent_name == "RESTAURANT":
        response = perform_entity_extraction(query,["date","no_of_reservations","restaurant_name"])
        
        if entity_is_null(response) != 0:
            return "Please type the query with the following datapoints referenced:"+ entity_is_null(response)
        else:
            return "Booking Confirmed for "+response["restaurant_name"][0]+" "+"for date "+to_readable(response["date"])+" "+"with number of reservations "+str(response["no_of_reservations"])
    elif intent_name == "REMINDER":
        response = perform_entity_extraction(query,["date","time","activity"])
        if entity_is_null(response) != 0:
            return "Please type the query with the following datapoints referenced:"+ entity_is_null(response)
        else:
            return response["activity"]+" added to Reminder List! Reminding you on "+to_readable(response["date"])+" at "+response["time"]
    