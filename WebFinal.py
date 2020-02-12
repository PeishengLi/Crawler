from selenium import webdriver
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
# create a empty database with two columns
theframe = pd.DataFrame(columns = ["text", "target"])
#setting browser for selenium
browser = webdriver.Chrome("/Users/peishengli/Downloads/chromedriver")
#use the browser to get data from target website
browser.get("https://steamcommunity.com/app/578080/reviews/?browsefilter=toprated&snr=1_5_100010_")
#scroll down with selenium
for i in range(0, 25):
    browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(1)

#use beautiful soup to parse the website
url = browser.page_source

soup = BeautifulSoup(url )

reviews = soup.find_all('div', {'class': 'apphub_CardContentMain'})
for review in reviews:
    temp = review.find("div", {"class": "vote_header"})
    title = review.find('div', {'class': 'title'}).text
    comment = review.find('div', {'class': 'apphub_CardTextContent'})
    thecomment = comment.text.strip().split("\n")[1]
    tempframe = pd.DataFrame({"text": [thecomment.replace("\t","").replace("the","").replace("is","").lower()], "target": [title]})
    theframe = theframe.append(tempframe, ignore_index = True)
    #print(title)
    #print(comment.text.strip().split("\n")[1])
    #print(tempframe)
    #print("____________________________________________________________")
    
print(theframe.head)
theframe.to_csv("/Users/peishengli/Desktop/Root/steamdata")
# Analyse the data parsed from steam
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB

X_text = theframe["text"]
Y = theframe["target"]
count_vectorizer = CountVectorizer(ngram_range=(1,2))
count_vectorizer.fit(X_text)
X = count_vectorizer.transform(X_text)
logistic_regression = ComplementNB()
aucs_log = cross_val_score(logistic_regression, X, Y, scoring="accuracy", cv=5)

print("\nAccuracy of our classifier(2-gram) is " + str(round(np.mean(aucs_log), 3))+"\n")


for i in theframe.groupby("target"):
    t = i[1].shape[0]
    print(i[0], t)
    
