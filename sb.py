import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,SpatialDropout1D,LSTM,Dropout,Activation,Bidirectional


trainDF = pd.read_csv("/content/drive/My Drive/Colab Notebooks/train_tweets.csv")
testDF = pd.read_csv("/content/drive/My Drive/Colab Notebooks/test_tweets.csv")

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


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(trainDF['tweet'].values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(trainDF['tweet'].values)
X = sequence.pad_sequences(X,maxlen=15,padding='post')
Y = trainDF['label']

# print (Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=42)

model = Sequential()
model.add(Embedding(len(word_index)+1, 100, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=64,epochs = 2,validation_split = 0.1)
score,acc = model.evaluate(X_test,Y_test)

print (score)
print (acc)

for i in range(len(testDF['tweet'])):
    testDF['tweet'].values[i] = preprocess(testDF['tweet'].values[i])

X = tokenizer.texts_to_sequences(testDF['tweet'].values)
X = sequence.pad_sequences(X,maxlen=15,padding='post')
pred = model.predict(X)
rounded = [round(x[0]) for x in pred]
y_pred = np.array(rounded,dtype='int64')
final_result = pd.DataFrame({'id':testDF['id'],'label':y_pred})
final_result.to_csv('/content/drive/My Drive/Colab Notebooks/output.csv',index=False)