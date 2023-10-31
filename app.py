import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from langdetect import detect
from translate import Translator

# Téléchargement des ressources linguistiques
nltk.download('punkt')
nltk.download('stopwords')

# Initialisation du stemmer
ps = PorterStemmer()

# Chargement du modèle TF-IDF et du modèle de classification
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Plateforme Web de Classification des SMS et Mails")
st.markdown("Conçue par Lucien TITO, Dev AI")

st.header("Classification de Messages")
st.write("Cette application permet de classifier les messages comme spam ou non spam.")

# Zone de saisie de texte
input_sms = st.text_area("Entrez votre message (dans n'importe quelle langue)")

if st.button('Prédire'):
    if len(input_sms) < 3:  # Ajoutez une limite minimale de 3 caractères pour la détection de la langue
        st.warning("Le texte est trop court pour détecter la langue.")
    else:
        # 1. Détection de la langue
        detected_language = detect(input_sms)

        # 2. Traduction en anglais si ce n'est pas en anglais
        if detected_language != 'en':
            translator = Translator(to_lang='en', from_lang=detected_language)
            translation = translator.translate(input_sms)
            input_sms = translation

        # 3. Prétraitement du texte
        transformed_sms = input_sms.lower()
        transformed_sms = nltk.word_tokenize(transformed_sms)
        transformed_sms = [i for i in transformed_sms if i.isalnum()]
        transformed_sms = [i for i in transformed_sms if i not in stopwords.words('english') and i not in string.punctuation]
        transformed_sms = [ps.stem(i) for i in transformed_sms]
        transformed_sms = " ".join(transformed_sms)

        # 4. Vectorisation et prédiction en utilisant le modèle en anglais
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # 5. Affichage du résultat
        if result == 1:
            st.error("Ce message est classé comme SPAM")
        else:
            st.success("Ce message n'est pas classé comme SPAM")
