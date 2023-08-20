# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:52:58 2023

@author: Mario Bagus Prasetya/6181801071
Kode untuk melakukan prediksi sentimen pada halaman website
"""

import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import nltk
import re
import Eksplorasi_Data as ed

import pandas as pd

def import_model_3label():
    import pickle5
    token = pickle5.load(open('other/model_vektor_3label.sav','rb'))
    model_load = pickle5.load(open('other/model_bilstm_1mei_10epoch.pkl','rb'))
    return [token,model_load]

def import_model_2label():
    import pickle5
    token = pickle5.load(open('other/model_vektor_2label.sav','rb'))
    model_load = pickle5.load(open('other/model_LSTM_80persen_5Mei_epoch20.pkl','rb'))
    return [token,model_load]

def tokenisasi(teks):
    token = nltk.tokenize.word_tokenize(teks) 
    return token
    
def get_encode(x,token):
    x = token.texts_to_sequences(x)
    x = pad_sequences(x,maxlen=250,padding="post")
    return x

def cleaning(teks):
    x = [teks]
    #casefold
    x= x[0].lower() 
    #cleaning
    x = re.sub(r'\d+', '', x)
    x = re.sub(r'[^\w\s]', ' ', x)
    x = re.sub(' +', ' ', x)
    return x

def stopwordRemover(teks):
    stop_factory = StopWordRemoverFactory()
    data = stop_factory.get_stop_words()
    data_sw = []
    negasi = pd.read_csv('other/negatingword.txt',header=None)
    list_negasi =negasi[0].tolist()
    for sw in data:
        if sw not in list_negasi:
            data_sw.append(sw)
    tambahan = ['makan','tempat']
    data_sw.extend(tambahan)
    bukan_stopword = []
    for kata in (tokenisasi(teks)):
            if kata not in data_sw:
                bukan_stopword.append(kata)
    sentence = ' '.join(bukan_stopword)
    return sentence

def stemming(teks):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    teks = stemmer.stem(teks)
    return teks

def sentiment_color(sentiment):
     if sentiment == "Positif" or sentiment == "positif":
         return "background-color:#4dff4d; color: white;font-weight: bold;"
     elif sentiment == "Negatif" or sentiment == "negatif":
         return "background-color: #ff3300; color: white;font-weight: bold;"
     else:
         return "background-color: #ffbf00; color: white;font-weight: bold;"    
     
def predict_satu_teks(x,token,model_load,jmlh_label):
    #hapus angka dan tandabaca
    x = cleaning(x)
    print(x)
    #stopword
    x = stopwordRemover(x)
    #stemming
   # x = stemming(x)
    print(x)
    x = [x]
    x = get_encode(x,token)
    #predict model
    label = ""  
    #lakukan prediksi model(model_load dari parameter function)
    predictX = model_load.predict(x)
    classes_x = np.argmax(predictX,axis=1)
    #jika jumlah label 3 maka ubah hasil prediksi (-1,0,1) 
    #menjadi negatif, netral atau positif
    if(jmlh_label == 3):
        if(classes_x==0):
            label = "Netral"
        elif(classes_x==1):
            label = "Positif"
        else:
            label = "Negatif"
    #jika jumlah label 2 maka ubah hasil prediksi (0,1) 
    #menjadi negatif atau positif
    elif(jmlh_label == 2):
        if(classes_x==0):
            label = "Negatif"
        elif(classes_x==1):
            label = "Positif"
    return label,predictX

def predict_file(jumlah_label,file):
    if(jumlah_label==3):        
        token,model_load = import_model_3label()
    elif(jumlah_label==2):
        token,model_load = import_model_2label()    
    df_hasil ={}
    list_review = []
    list_label = []
    for teks in file['review']:
        list_review.append(teks)
        label,predictX = predict_satu_teks(teks,token,model_load,jumlah_label)
        list_label.append(label)
        #st.write("Teks : ", teks)
        #st.write(predict_satu_teks(teks))
    df_hasil = {'review':list_review, 'label':list_label}
    return pd.DataFrame(df_hasil)
    #print(file['review'])
  
def tampilkan_prediksi_teks():
    input_text = st.text_input('Masukan text:')
    st.write("Text yang ingin dianalisis : ", input_text)
    choose_model = st.selectbox("Pilih model untuk prediksi",("BiLSTM 2 label","BiLSTM 3 label"),key=3)
    prediksi_1 = st.button("Prediksi Sentiment",key = 1)
    if(prediksi_1):
        if(input_text != ""):
            if(choose_model == "BiLSTM 3 label"): 
                token,model_load = import_model_3label()
                label, predictX = predict_satu_teks(input_text,token,model_load,3)
            elif(choose_model == "BiLSTM 2 label"):
                token,model_load = import_model_2label()
                label, predictX = predict_satu_teks(input_text,token,model_load,2)
            if(label == "Netral"):
                print("netral")
                st.markdown("Hasil analisis: **:orange[Netral]**")
                st.write("Kepercayaan : " + str(int(np.max(predictX)*100))+"%")           
            elif(label == "Positif"):
                print("positif")
                st.markdown("Hasil analisis: **:green[Positif]**")
                st.write("Kepercayaan : " + str(int(np.max(predictX)*100))+"%")            
            else:
                print("negatif")
                st.markdown("Hasil analisis: **:red[Negatif]**")
                st.write("Kepercayaan : " + str(int(np.max(predictX)*100))+"%")
        else:
            st.warning('Masukan Teks Terlebih Dahulu!')
            
def tampilkan_prediksi_file_teks():
    try:
        uploaded_file = st.file_uploader("Choose a file")
        if(uploaded_file):
           #st.write('File masuk: ', uploaded_file)   
           jenis_file = uploaded_file.name.split('.')[1]
           print(uploaded_file.name.split('.')[1])
           if(jenis_file == "xlsx"):
               df = pd.read_excel(uploaded_file)
               df.columns=['review']
           elif(jenis_file == "csv"):
               df = pd.read_csv(uploaded_file)
               df.columns=['review']
           
           #print(df)
           with st.expander("Lihat data"):
               st.dataframe(df)
           choose_model = st.selectbox("Pilih model untuk prediksi",("BiLSTM 2 label","BiLSTM 3 label"),key=4)
           prediksi_2 = st.button("Prediksi Sentiment",key = 2)
           if(prediksi_2):
               if(choose_model=="BiLSTM 3 label"):
                   df_labeled = predict_file(3,df)
                   print(df_labeled)
               elif(choose_model=="BiLSTM 2 label"):
                   df_labeled = predict_file(2,df)
               col1, col2= st.columns([50, 50])
               with col1:
                   print(df_labeled)
                   st.dataframe(df_labeled[["review", "label"]].style.applymap(
                                sentiment_color, subset=["label"])) 
                   print(df_labeled.label.value_counts())
               with col2:
                   sentiment_plot = ed.piechart_sentiment(df_labeled)
                   sentiment_plot.update_layout(height=400, title_x=0.5)
                   st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
    except:
        st.warning("Anda memasukan jenis File yang salah. Harap masukan file excel atau csv!")