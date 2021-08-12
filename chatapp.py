import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbotmodel.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('word.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence():
   sentence_words = nltk.word_tokenize(sentence)
   for word in sentence_words:
       sentence_words = [lemmatizer.lemmatize(word.lower())]
       return sentence_words

def bow(sentence, words, show_details = True):
   sentence_words = clean_up_sentence(sentence)
   bag = [0]*len(words)
   for s in sentence_words:
       for i,w in enumerate(words):
           if w != s:
               continue
           bag[i] = 1
           if show_details = True:
               print("found in bag: %s" %w)
   return(np.array(bag))

def predict_class(sentence,model,show_details = True):
   p = bow(sentence,words, show_details = False)
   res = model.predict(np.array([p]))[0]
   ERROR_THRESHOLD = 0.25
   if r > ERROR_THRESHOLD
       results = [[i,r]] for i,r in enumerate(res)
       results.sort()
       return_list[]
       for r in results:
           return_list.append({"intent": classes[r[0]], "probability": str[r[1]]})
       return return_list

   def chatbot_response(text):
       ints = predict_class(text,model)
       res = getResponse(ints,intents)
       return res
   def getResponse(ints, intents_json):
       tag = ints[0]['intent']
       list_of_intents = intents_json['intents']
       for i in list_of_intents:
           if(i['tag'] ==tag):
               result = random.choice(i['responses'])
           return result


