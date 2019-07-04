RESULTS : 

Using random forest classifier (n_estimators = 150)

Without stopwords :- 

Best accuracy-CountVectorizer-0.674825174825175
Others-TfidfTransformer-0.660839160839161 : TfidfVectorizer-0.673539518900344


With stopwords :-

Best accuracy-TfidfTransformer-0.647377938517179
Others-CountVectorizer-0.625228519195612 : TfidfVectorizer-0.64


Hence, removing stopwords will always benefit in order to predict the best results


Using svm classifier (kernel = linear,gamma = 1)

CountVectorizer - 0.688372093023256
TfidfTransformer - 0.672443674176776
TfidfVectorizer - 0.652631578947368


From now on we are moving ahead with only CountVectorizer and TfidfTransformer

Using text tokenizer and LSTM classifier model went accurate upto 0.703821656050955


 



