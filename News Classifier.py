#Purpose: The below code aims to classify news headlines or news articles into different categories using the Naive Bayes classifier.
#A function is defined to accept news from the user to predict its category and another function to groups similar categories for better readability.

#Load the required libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#Load the dataset
news = fetch_20newsgroups()
news.target_names

#Define these categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
              'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 
              'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
              'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

#Training the data on these categories
news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle='true')
#Testing the data for these categories 
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle='true')

#Creating a model using Multinomial Naive Bayes.Pipeline is used to pump the weights created by the text-vectorizer into the MultinomialNB classifier.
text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
#Training the model with Train data
text_clf.fit(news_train.data, news_train.target)
#Predicting the label for test data
predicted_label = text_clf.predict(news_test.data)

#Printing the model accuracy
print("Accuracy Score of the model:", round(metrics.accuracy_score(news_test.target, predicted_label),2))

#Creating confusion matrix and heatmap
mat = metrics.confusion_matrix(news_test.target,predicted_label)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
            , xticklabels=news_train.target_names 
            , yticklabels=news_train.target_names,
              annot_kws={"size": 8, "color": "white", "weight": "bold"}) #Annotation text customization

#Plotting heatmap of confusion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix')
plt.show()

#Function to predict the news category on new data based on trained model
def predict_newscategory(s, train=news_train, model=text_clf):
    pred= text_clf.predict([s])
    return news_train.target_names[pred[0]]

#Function to group the similar categories and return a generalized name
def rename_category(c):
    if c in ("comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x","sci.crypt","sci.electronics"):
        return "Technology and Computing"
    elif c in ("rec.autos","rec.motorcycles"):
        return "Automobiles"
    elif c in ("rec.sport.baseball","rec.sport.hockey"):
        return "Sports"
    elif c == "sci.med":
        return "Medicine"
    elif c == "sci.space":
        return "Space"
    elif c in ("talk.politics.guns","talk.politics.mideast","talk.politics.misc"):
        return "Politics"
    elif c in ("soc.religion.christian","talk.religion.misc","soc.religion.christian"):
     return "Philosophy and Religion"
    elif c == "misc.forsale":
     return "Marketplace and Sales"
        

#Taking multiple news headlines or news articles to predict the news category
while(True):
    s = input("Enter the news you want to classify.(Press Q to end)\n")
    if s=="Q":
        break
    print("The news belongs to the category: ",rename_category(predict_newscategory(s)))
