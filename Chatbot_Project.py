import random
import warnings
import json
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import spacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity 
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from IPython import get_ipython

# In[46]:


warnings.filterwarnings('ignore') 


# In[47]:


# Launching spacy
nlp = spacy.load("en_core_web_sm")


# In[48]:


# Mounting the google drive
#from google.colab import drive
#drive.mount('/content/drive')


# # **Intent Classification**

# ## Data Preparation

# In[49]:


# loading the intent file
data_file = open('intents.json').read()
intents_patterns = json.loads(data_file)


# In[50]:


# preparing the data to be ready for machine learning
data = []
for intent in intents_patterns['intents']:
  for pattern in intent['patterns']:
    data.append((pattern , intent['tag']))

tags = [intent['tag'] for intent in intents_patterns['intents']]

intent_df= pd.DataFrame(data = data, columns = ['pattern', 'intent'])
intent_df


# ## Pattern Preprocessing

# In[51]:


# a function to normalize the text
def normalize_doc(text):
  # lower case and remove special characters\whitespaces
  text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  text = text.lower()
  # lemmatization
  #doc = nlp(text)
  #lemmas = [token.lemma_ for token in doc]
  # re-create document from lemmas
  #text = ' '.join(lemmas)
  return text


# In[52]:


intent_df['pattern_normalized'] = intent_df['pattern'].apply(normalize_doc)


# In[53]:


intent_df.head()


# ## Feature Engineering

# ### TFIDF

# In[54]:


# Extract the feature column and the target variable
X = intent_df['pattern_normalized']
y = intent_df['intent']

# Perform Splitting
x_train , x_test, y_train , y_test = train_test_split(X, y, test_size = 0.25, random_state = 111, stratify = y)


# In[55]:


# instantiating the tfidf vectorizer
#tf_ml = TfidfVectorizer()

# matrix for training
#X_train_tfidf = tf_ml.fit_transform(x_train)

# matrix for testing
#X_test_tfidf = tf_ml.transform(x_test)


# ### BERT

# In[56]:


#get_ipython().system('pip install -U sentence-transformers')


# In[57]:


from sentence_transformers import SentenceTransformer
# instantiating the embedder 
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


# In[58]:


# vectorizing the corpus using embeddings
corpus_embeddings = embedder.encode(X)
corpus_embeddings.shape

# splitting the embeddings data
x_train_emb , x_test_emb , y_train_emb , y_test_emb = train_test_split(corpus_embeddings, y, test_size = 0.25, random_state = 111, stratify = y)


# ## Baseline Model for Intent Classification (Cosine Similarity)

# In[59]:


# Cosine similarity method using TFIDF vectorization
# y_pred_cos=[]
# for i in range(X_test_tfidf.shape[0]):
#   scores = cosine_similarity(X_train_tfidf, X_test_tfidf[i])
#   y_pred_cos.append(y_train.iloc[np.argmax(scores)])

# # Accuracy of cos
# accuracy_cos = accuracy_score(y_test,y_pred_cos)*100
# accuracy_cos


# In[60]:


# Cosine similarity method using Bert vectorization
# y_pred_bert=[]
# for i in range(x_test_emb.shape[0]):
#   scores = cosine_similarity(x_train_emb, x_test_emb[i].reshape(1, -1))
#   y_pred_bert.append(y_train_emb.iloc[np.argmax(scores)])

# # Accuracy of bert
# accuracy_bert = accuracy_score(y_test_emb,y_pred_bert)*100
# accuracy_bert


# ## Machine Learning Approach

# ### Models Definition

# In[61]:


# a function to train and test the model
# def train_test_model(model, x_train, x_test, y_train, y_test):
#   # training the model
#   model.fit(x_train, y_train)
#   # predicting the test data
#   y_pred = model.predict(x_test)
#   acc = accuracy_score(y_test, y_pred)
#   return acc 


# In[62]:


# # instanstiating several models
# lr_tfidf = LogisticRegression(max_iter=10000)
# svm_tfidf = SVC()
# clf_tfidf = DecisionTreeClassifier(random_state=0)
# ens_clf_tfidf = RandomForestClassifier(10000)
# all_models_tfidf = [lr_tfidf, svm_tfidf, clf_tfidf, ens_clf_tfidf]


# ### ML Models Using TFIDF

# #### Running Models

# In[63]:


# def train_tfidf():
#   model_performance_tf_idf = {}
#   for model in all_models_tfidf:
#     model_performance_tf_idf[model] = (train_test_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test))
#   return model_performance_tf_idf


# # In[64]:


# train_tfidf()


# # #### Optimizing Models

# # In[65]:


# lr = LogisticRegression()


# In[66]:


# param_grid = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'max_iter': [7000,10000]}]
# grid_search = GridSearchCV(lr, param_grid, verbose=2, n_jobs=-1)
# grid_search.fit(X_train_tfidf , y_train)


# # In[67]:


# lr = LogisticRegression(max_iter= grid_search.best_params_['max_iter'], penalty= grid_search.best_params_['penalty'], solver= grid_search.best_params_['solver'])


# # In[68]:


# train_test_model(lr, X_train_tfidf, X_test_tfidf, y_train, y_test)


# ### ML Models Using Embeddings (BERT)

# #### Running Models

# In[69]:


# instanstiating several models
# lr_emb = LogisticRegression(max_iter=10000)
# svm_emb = SVC()
# clf_emb = DecisionTreeClassifier(random_state=0)
# ens_clf_emb = RandomForestClassifier(10000)
# all_models_emb = [lr_emb, svm_emb, clf_emb, ens_clf_emb]


# In[70]:


# def train_emb():
#   model_performance_embeddings = {}
#   for model in all_models_emb:
#     model_performance_embeddings[model] = (train_test_model(model, x_train_emb, x_test_emb, y_train_emb, y_test_emb))
#   return model_performance_embeddings


# In[71]:


# train_emb()


# #### Optimizing Models

# In[73]:


# param_grid = [{'penalty': ['l2', 'none'], 'solver' : ['liblinear', 'sag', 'saga'],'max_iter': [7000,10000]}]

# grid_search = GridSearchCV(lr, param_grid, verbose=2, n_jobs=-1)
# grid_search.fit(x_train_emb , y_train_emb)


# # In[74]:


# lr = LogisticRegression(max_iter= grid_search.best_params_['max_iter'], penalty=grid_search.best_params_['penalty'], solver= grid_search.best_params_['solver'])


# # In[75]:


# train_test_model(lr, x_train_emb, x_test_emb, y_train_emb, y_test_emb)


# ## Neural Network Approach

# ### NN Preprocessing (Encoding Intent)

# In[76]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

label_decoder = dict(zip(y, dummy_y))


# ### Using TFIDF

# In[77]:


# tf_neural = TfidfVectorizer()
# tf_matrix_keras = tf_neural.fit_transform(X)
# dense_features = tf_matrix_keras.todense()


# In[78]:


# Create model_tfidf - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
# model_tfidf = Sequential()
# model_tfidf.add(Dense(128, input_shape=(dense_features.shape[1],), activation='relu'))
# model_tfidf.add(Dropout(0.5))
# model_tfidf.add(Dense(64, activation='relu'))
# model_tfidf.add(Dropout(0.5))
# model_tfidf.add(Dense(len(dummy_y[0]),activation='softmax'))

# # Compile model_tfidf. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model_tfidf
# sgd = SGD(learning_rate = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model_tfidf.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# #fitting and saving the model_tfidf 
# hist = model_tfidf.fit(dense_features.astype('float32'), dummy_y, epochs=400, verbose=1)
# #model_tfidf.save('chatbot_model_tfidf.h5', hist)

# print("model_tfidf created")


# ### Using Embeddings (BERT)

# In[79]:


# Create model_emb - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model_emb = Sequential()
model_emb.add(Dense(128, input_shape=(768,), activation='relu'))
model_emb.add(Dropout(0.5))
model_emb.add(Dense(64, activation='relu'))
model_emb.add(Dropout(0.5))
model_emb.add(Dense(len(dummy_y[0]),activation='softmax'))

# Compile model_emb. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model_emb
sgd = SGD(learning_rate = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_emb.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model_emb 
hist = model_emb.fit(corpus_embeddings, dummy_y, epochs=400, verbose=1)
#model_emb.save('chatbot_model_emb.h5', hist)

# print("model_emb created")


# ### Testing NN

# In[80]:


# trial
# testing_test = "what is word embeddings"
# testing the tfidf approach
# np.argmax(model_tfidf.predict((tf_neural.transform([testing_test])).todense()))
# testing the embedding approach
#  index = np.argmax(model_emb.predict(embedder.encode(testing_test).reshape(1,-1)))


# In[81]:


# getting the label
# for key, value in label_decoder.items():
#   if np.argmax(value) == index:
#     print(key)


# # **Information Retrieval (IR)** 

# ## Data Preparation

# In[82]:

# Reading the NLP corpus
documnets_names = ["General.txt","Feature Engineering.txt","Langauge Models.txt", "Sequence Labeling.txt", 
                   "Text Classification and SA.txt", "Text Preprocessing.txt", "Text Similarity, Clustering and IR.txt",
                   "Topic Modeling and Text Sum.txt"]
corpus = ''                 
for name in documnets_names:
  document = open( name ,'r', encoding="utf8")
  document = document.read()
  corpus = corpus + document + " "

# NLP Corpus Preprocessing

# removing wikipedia references (many texts are from wiki), \n, and :
def remove_references_newlines(text):
    text = re.sub(r'\[\d+\]', '',text)
    text = re.sub(r'\n', '',text)
    text = re.sub(r':', ',',text)
    return text

corpus = remove_references_newlines(corpus)

# Sentence segmentation
doc = nlp(corpus)
sentences = [sent.text for sent in doc.sents]

# Saving sentences in a dataframe
df = pd.DataFrame()
df["sentences"] = sentences


def normalize_nlp_document(text, normMethod):
    # lower case and remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    text = text.lower()
    # tokenize document
    tokens = nltk.word_tokenize(text)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    # Stemming 
    if normMethod == "stemming":
      porter = PorterStemmer()
      stems = [porter.stem(token) for token in filtered_tokens]
      # re-create document from filtered tokens
      text = ' '.join(stems)
      return text
    # Lemmatization
    elif normMethod == "lemmatization":
       text = ' '.join(filtered_tokens)
       doc = nlp(text)
       lemmas = [token.lemma_ for token in doc]
       # re-create document from lemmas
       text = ' '.join(lemmas)
       return text


# Adding normalized sentence columns
df['stemming normalized sentences'] = df['sentences'].apply(normalize_nlp_document, normMethod='stemming')
df['lemmatization normalized sentences'] = df['sentences'].apply(normalize_nlp_document, normMethod='lemmatization')


#IR Using Embeddings 
# instantiating the embedder 
#embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
ir_embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
# vectorizing the corpus using embeddings
stemming_corpus_embeddings = ir_embedder.encode(df['stemming normalized sentences'])
lemmatization_corpus_embeddings = ir_embedder.encode(df['lemmatization normalized sentences'])

def retrieve_using_emb(query, normMethod="stemming"):
  if normMethod == "stemming":
    col = df['stemming normalized sentences']
    corpus_embeddings = stemming_corpus_embeddings
  elif normMethod == "lemmatization":
    col = df['lemmatization normalized sentences']
    corpus_embeddings = lemmatization_corpus_embeddings

  # Query preprocessing
  query = normalize_nlp_document(query,normMethod)
  mat = ir_embedder.encode([query])
 
  # Comparison
  query_doc_sim = cosine_similarity(mat,corpus_embeddings)[0]
  top_rel =  (-query_doc_sim).argsort()[0]
  print(df['sentences'].iloc[top_rel])
  #random.randint(0, 1)

  # looking into the neighbors of the match
  if top_rel > 2 and top_rel < corpus_embeddings.shape[0]:
    neighbor = cosine_similarity(ir_embedder.encode([col.iloc[top_rel]]),corpus_embeddings[top_rel-3:top_rel+4])[0]
    #print(neighbor)
    i = 3
    neighbor_tuples = []
    for n in neighbor:
      neighbor_tuples.append((top_rel-i, n))
      if len(neighbor_tuples)!=len(neighbor):
        i=i-1
    info = ""
    for t in neighbor_tuples:
      if t[1] > 0.55:
        info = info + df['sentences'].iloc[t[0]] + " "
  else:
    info = df['sentences'].iloc[top_rel]

  return info


intent_responses_dic = {}
for intent in intents_patterns['intents']:
  intent_responses_dic[intent['tag']] = intent['responses'] 

def get_response(message):
  norm_message = normalize_doc(message)
  index = np.argmax(model_emb.predict(embedder.encode(norm_message).reshape(1,-1)))
  # getting the label
  for key, value in label_decoder.items():
    if np.argmax(value) == index:
      if key != "Natural Language Processing":
        response = random.choice(intent_responses_dic[key])
      elif key == "Natural Language Processing":
        response = retrieve_using_emb(message)
  return response


########################################################################################
###################################### THE END #########################################
########################################################################################