import streamlit as st
from transformers import BertTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer #to stem words
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#functions
@st.cache_resource()
def get_model():
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  p = pipeline("text2text-generation",model=model,tokenizer=tokenizer)
  sent_tokenizer = AutoTokenizer.from_pretrained('akhooli/xlm-r-large-arabic-sent')
  sent_model = AutoModelForSequenceClassification.from_pretrained("akhooli/xlm-r-large-arabic-sent")

  return tokenizer, model, p,sent_tokenizer,sent_model
  


def get_color(num):
    return "#98FB98" if num<0.4 else "#FAFAD2" if num<0.6 else "#FA8072"

######################################################################
##########################SENTIMENT ANALYSIS##########################
######################################################################




punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

# Arabic stop words with nltk
stop_words = stopwords.words('arabic')

arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def preprocess(text):

    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)

    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)
    text = re.sub('[A-Za-z0-9]',' ',text)

    #remove longation (longation basically is a form of arabic diacritics )
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    #next creating a list of substrings
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text
def predict(text ,sent_tokenizer,sent_model):
  text = preprocess(text)
  tokened_str = sent_tokenizer(text, return_tensors='pt')
  propabiility = softmax(sent_model(**tokened_str).logits.data.numpy())[0]
  return propabiility[0],propabiility[1],propabiility[2]
##################################################
###################### MAIN ######################
##################################################
#variables 
model_name="malmarjeh/mbert2mbert-arabic-text-summarization"

if 'resList' not in st.session_state:
    st.session_state['resList'] = []
if "model" not in st.session_state.keys():
    tokenizer, model, pipeline,sent_tokenizer,sent_model = get_model()
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["pipeline"] = pipeline
    st.session_state["sent_tokenizer"] = sent_tokenizer
    st.session_state["sent_model"] = sent_model

# model = st.session_state["model"] 

#body
# tokenizer, model, pipeline,sent_tokenizer,sent_model = get_model()
st.title("Arabic Text Summarization 💬")
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

if user_input and button :
    message_text =st.session_state["pipeline"](user_input,
            pad_token_id=st.session_state["tokenizer"].eos_token_id,
            num_beams=3,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=1.0,
            no_repeat_ngram_size = 3)[0]['generated_text']
    a,negative,c=predict(user_input,st.session_state["sent_tokenizer"],st.session_state["sent_model"])
    color =get_color(negative)
    st.session_state["resList"].append(f'''
                                       <p style='background-color: {color}; padding: 10px; margin: 5px; border-radius: 5px;'>
                                       {message_text}<br> Negativity:{negative:0.2f} &emsp;Positivity:{a:0.2f}&emsp;Neutral:{c:0.2f}</p>
                                       ''')

    for i in st.session_state['resList']:
      st.markdown(i,unsafe_allow_html=True)


