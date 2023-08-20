# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:35:56 2023

@author: USER
"""
import streamlit as st
from Prediksi_Sentimen import tampilkan_prediksi_teks
from Prediksi_Sentimen import tampilkan_prediksi_file_teks
import Eksplorasi_Data as ed
import pandas as pd
from Prediksi_Sentimen import sentiment_color

st.set_page_config(
    page_title="Sentimen Analisis", page_icon="",layout="wide", 
)
adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

#sidebar
with st.sidebar:
    st.title("Website Sentiment Analyzer")
    page = st.selectbox("Pilih halaman",("Predict Page","Explore Page"))
    st.write("""Created by Mario Bagus Prasetya""")
    
#pageprediksi
if(page == "Predict Page"):
    st.title("Prediksi Sentimen Wisata Kuliner dan Belanja di Kota Bandung")
    tab1, tab2 = st.tabs(["Input 1 teks", "Input File"])
    with tab1:
       tampilkan_prediksi_teks()
    with tab2:
       tampilkan_prediksi_file_teks()
#page EDA
elif(page == "Explore Page"):        
    st.title("Eksplorasi Data")
    tab1,tab2,tab3,tab4 = st.tabs(["All data", "Positive", "Negative", "Netral"])
    df = pd.read_csv('other/data_Skripsi_fix2.csv')
    #all data
    with tab1:
        col1, col2 = st.columns([50, 50])
        #piechart
        with col1:
            sentiment_plot = ed.piechart_sentiment(df)
            sentiment_plot.update_layout(height=400, title_x=0.5)
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        #barplot
        with col2:
            st.pyplot(ed.barplot_sentiment(df,'#1a75ff',"all"))
        col3, col4 = st.columns([60, 40])
        #datafrane
        with col3:
            import pandas as pd
            # show the dataframe containing the tweets and their sentiment
            st.dataframe(
                df[["review", "label"]].style.applymap(
                    sentiment_color, subset=["label"]),
                height=550)
       #wordcloud
        with col4:
           ed.wordcloud(df)
    #positive
    with tab2:
       positif = df[df['label']=='positif']
       col1, col2= st.columns([50, 50])
       #piechart
       df = pd.read_csv('other/data_Skripsi_fix2.csv') 
       with col1:
           color = '#4dff4d'
           st.pyplot(ed.barplot_sentiment(positif,color,"positif"))
       #barplot
       with col2:
           color = '#4dff4d'
           st.pyplot(ed.barplot_sentimentWithLexicon(positif,color,"positif"))
       col3, col4 = st.columns([60, 40])
       #datafrane
       with col3:
           import pandas as pd
           # show the dataframe containing the tweets and their sentiment
           st.dataframe(
               positif[["review", "label"]].style.applymap(
                   sentiment_color, subset=["label"]),
               height=550)
      #wordcloud
       with col4:
          ed.wordcloud(positif)
    #negative
    with tab3:
       negatif = df[df['label']=='negatif']
       col1, col2= st.columns([50, 50])
       #piechart
       df = pd.read_csv('other/data_Skripsi_fix2.csv') 
       with col1:
           color = '#ff3300'
           st.pyplot(ed.barplot_sentiment(negatif,color,"negatif"))
       #barplot
       with col2:
           color = '#ff3300'
           st.pyplot(ed.barplot_sentimentWithLexicon(negatif,color,"negatif"))
       col3, col4 = st.columns([60, 40])
       #datafrane
       with col3:
           import pandas as pd
           # show the dataframe containing the tweets and their sentiment
           st.dataframe(
               negatif[["review", "label"]].style.applymap(
                   sentiment_color, subset=["label"]),
               height=550)
      #wordcloud
       with col4:
          ed.wordcloud(negatif)
    #netral
    with tab4:
        netral = df[df['label']=='netral']

        #piechart
        df = pd.read_csv('other/data_Skripsi_fix2.csv') 
        color = '#ffbf00'
        st.pyplot(ed.barplot_sentiment(netral,color,"netral"))

        col1, col2 = st.columns([60, 40])
        #datafrane
        with col1:
            import pandas as pd
            # show the dataframe containing the tweets and their sentiment
            st.dataframe(
                netral[["review", "label"]].style.applymap(
                    sentiment_color, subset=["label"]),
                height=550)
       #wordcloud
        with col2:
           ed.wordcloud(netral)
