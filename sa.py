import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

trainDF = pd.read_csv("train_tweets.csv")
testDF = pd.read_csv("test_tweets.csv")

trainDF = trainDF.drop('id',axis = 1)
trainDF = trainDF.sample(frac=1)
stops = set(stopwords.words('english'))

def preprocess (sent):
    sent = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])","",sent)
    lis = [word for word in sent.split() if word not in stops]
    sent = " ".join(lis)
    return sent

for i in range(len(trainDF['tweet'])):
    trainDF['tweet'].values[i] = preprocess(trainDF['tweet'].values[i])

X_train,X_test,Y_train,Y_test = train_test_split(trainDF['tweet'],trainDF['label'],test_size = 0.3,random_state=42)

counter = CountVectorizer()
transformer = TfidfTransformer(norm='l2',sublinear_tf = True)

x_train_count = counter.fit_transform(X_train)
x_train_trans = transformer.fit_transform(x_train_count)

x_test_count = counter.transform(X_test)
x_test_trans = transformer.transform(x_test_count)

model1 = svm.SVC(kernel='linear', gamma=1)
model2 = svm.SVC(kernel='linear', gamma=1)
model1.fit(x_train_count,Y_train)
model2.fit(x_train_trans,Y_train)

pred_count = model1.predict(x_test_count)
pred_trans = model2.predict(x_test_trans)

print(accuracy_score(Y_test,pred_count))
print(accuracy_score(Y_test,pred_trans))

for i in range(len(testDF['tweet'])):
    testDF['tweet'].values[i] = preprocess(testDF['tweet'].values[i])

test_count = counter.transform(testDF['tweet'])
test_trans = transformer.transform(test_count)

pred_count = model1.predict(test_count)
pred_trans = model2.predict(test_trans)

final_result = pd.DataFrame({'id':testDF['id'],'label':pred_vect})
final_result.to_csv('output.csv',index=False)

