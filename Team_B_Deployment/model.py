import nltk
#nltk.download('vader_lexicon')
#nltk.download('punkt')
from sklearn.externals import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
class Models():
    def __init__(self):
        pass
    def convert(self,x):
        if x < -0.5:
            return -1
        elif x > 0.5:
            return 1
        else:
            return 0

    def SentimentModel(self,text):
        text=str(text)
        text=text.replace('\d+', '')
        text=text.replace(r'[^\w\s]+', '')
        text=text.replace(r'\^[a-zA-Z]\s+', '')
        text=text.lower()
        sid = SentimentIntensityAnalyzer()
        sentiment=sid.polarity_scores(text)
        score=self.convert(sentiment['compound'])
        return score

    def Stock_model(self,day,month):
        smodel=pickle.load(open('Saved_model/final_Model.sav','rb'))
        return smodel.predict([[int(day),int(month)]])[0]
        

        


