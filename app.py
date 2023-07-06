import pickle
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.corpus.reader import wordnet
import flask
from flask import Flask, render_template, request


def tokenize_lemma(text):
    tokens = nltk.word_tokenize(text)
    lemm = nltk.WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        tag = nltk.pos_tag([tok])[0][1][0]
        tag_dict = {
            "J": wordnet.ADJ,
            "R": wordnet.ADV,
            "N": wordnet.NOUN,
            "V": wordnet.VERB
        }
        tag_dict = tag_dict.get(tag, wordnet.NOUN)
        clean_tokens.append(lemm.lemmatize(tok, tag_dict))
    return clean_tokens

nltk.download(['punkt', 'averaged_perceptron_tagger'])

app=Flask(__name__)

#Depickling the model
clf=pickle.load(open('model.pkl','rb'))

#Depickling the count vect
vectorizer=pickle.load(open('cnt_vectorizer.pkl','rb'))

#Depickling the TFIDF vect
tfidf=pickle.load(open('tfidf_vect.pkl','rb'))

#Depickling label Encoder
label_encoder=pickle.load(open('label_en.pkl','rb'))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    #label encode and normalize the inputs
    features = request.form.values()

    output = clf.predict(tfidf.transform(vectorizer.transform(features)))
    print(f'Classfier output = {output}')
    tweet_sentiment = label_encoder.inverse_transform(output)
    print(f'decoded output = {tweet_sentiment}')
    if tweet_sentiment == 'positive':
        pred = 'POSITIVE'
    elif tweet_sentiment == 'negative':
        pred = 'NEGATIVE'
    else:
        pred = 'NEUTRAL'

    return render_template('index.html', prog = pred)


if __name__ == "__main__":
    app.run(debug=True) #create a flask local server