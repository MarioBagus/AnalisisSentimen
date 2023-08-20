# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:23:49 2023

@author: Mario Bagus Prasetya/6181801071
Kode untuk Visualisasi data pada website
"""
import plotly.express as px
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist
nltk.download('punkt')
#import kata positive
data_pos = pd.read_csv('other/positive.txt',header=None)
list_pos = data_pos[0].tolist()
#import kata negative
data_negative = pd.read_csv('other/negative.txt',header=None)
list_negative = data_negative[0].tolist()

def piechart_sentiment(df):
    # Menghitung jumlah masing-masing label
    sentiment_count = df["label"].value_counts()

    # Buat piechart
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
       
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        # warna untuk positif, negatif dan netral
        color_discrete_map={"positif": "#4dff4d", "negatif": "#ff3300","netral":'#ffbf00',"Positif": "#4dff4d", "Negatif": "#ff3300","Netral":'#ffbf00'},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig

def barplot_sentiment(df,colorBar,sentimen):
    a = df['review'].str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    hasil = pd.DataFrame(word_dist.most_common(15),columns=['Kata', 'Frekuensi'])
    name = hasil['Kata']
    price = hasil['Frekuensi']
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
     
    #Horizontal Bar Plot
    ax.barh(name, price,color=colorBar)

    #Menghapus axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
     
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
     
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
     
    #Menambahkan x, y gridlines
    ax.grid( color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
     
    # Show top values
    ax.invert_yaxis()
     
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 12, fontweight ='bold',
                 color ='grey')
     
    #Menambahkan Plot Title
    if(sentimen == "all"):
        ax.set_title('15 kata terbanyak didalam dataset',
                     loc ='center',fontsize = 20,fontweight ='bold' )
    else:
        tittle = '15 kata terbanyak didalam dataset berlabel ' + sentimen 
        ax.set_title(tittle,
                     loc ='center',fontsize = 20,fontweight ='bold' ) 
    return fig

def barplot_sentimentWithLexicon(df,colorBar,sentimen):
    a = df['review'].str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    word_withSentiment = {}
    if(sentimen == 'negatif'):
        for i in word_dist:
            if(i in list_negative):
                word_withSentiment[i]=word_dist[i]
    elif(sentimen == 'positif'):
        for i in word_dist:
            if(i in list_pos):
                word_withSentiment[i]=word_dist[i]
    word_withSentiment = FreqDist(word_withSentiment)
    hasil = pd.DataFrame(word_withSentiment.most_common(15),columns=['Kata', 'Frekuensi'])
    name = hasil['Kata']
    price = hasil['Frekuensi']
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
     
    #Membuat barplot menjadi horizontal
    ax.barh(name, price,color=colorBar)

    #Menghapus axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
     
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
     
    #Menambah padding antara axes dan labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
     
    #Menambah x, y gridlines
    ax.grid( color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
     
    ax.invert_yaxis()
     
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 12, fontweight ='bold',
                 color ='grey')
     
    # Menambahkan Plot Title
    if(sentimen == "all"):
        ax.set_title('15 kata terbanyak didalam dataset',
                     loc ='center',fontsize = 20,fontweight ='bold' )
    else:
        tittle = '15 kata '+ sentimen + ' terbanyak dalam label ' + sentimen
        ax.set_title(tittle,
                     loc ='center',fontsize = 20,fontweight ='bold' )
    return fig

def wordcloud(df):
    comment_words = ''
    stopwords = ['tidak','jadi','kalau','nya']
     
    #iterasi setiap review pada kolom review
    for val in df.review:
        # typecaste setiap val ke string
        val = str(val)
        # split value menjadi token
        tokens = val.split()    
        tokens_with_sw = []
        for i in tokens:
            if(i not in stopwords):
                tokens_with_sw.append(i)
        comment_words += " ".join(tokens_with_sw)+" "
     
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image                      
    plt.figure(figsize = (10, 10), facecolor = None)
    plt.title("Wordcloud",fontweight ='bold' )
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    #plt.show()
    st.pyplot(plt)
    return comment_words
