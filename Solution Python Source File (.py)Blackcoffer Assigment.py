#!/usr/bin/env python
# coding: utf-8

# # **Data Extraction and NLP**

# ## **Objective**
# 
#    * **The objective of this assignment is to extract textual data articles from the given URL and perform text analysis to             compute variables that are explained below.**

# ## **import necessary pacakages**
# 

# In[ ]:


#pip install pyphen


# In[3]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
import newspaper

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("averaged_perceptron_tagger")

import re
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import pyphen


# In[4]:



def extract_article_text(url):
    """
    this function Extract the Title & Article_Text
    this function only extract the Aticle Title & Article Text
    It should not extract the website header, footer, or anything other than the article text
    
    """
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
    except newspaper.ArticleException as e:
        print(f"Error occurred while extracting the article: {e}")
        return None, None

    title = article.title
    article_text = article.text

    return title, article_text


#<================================================================================================================================>

def save_article_to_folder(url_id, title, article_text):
    """
    this function ,
    For each of the articles, given in the input.xlsx file,
    extract the article text and save the extracted article in a text file with URL_ID as its file name
    """
    folder_name = f"{url_id}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    with open(os.path.join(folder_name, "extracted_article.txt"), "w", encoding="utf-8") as file:
        file.write(f"{title}\n\n")
        file.write(article_text)
        
        
#<================================================================================================================================>

def count_syllables(word):
    """
    this function count Syllables form articles
    """
    dic = pyphen.Pyphen(lang="en_US")
    return len(dic.inserted(word).split("-"))


#<================================================================================================================================>
#<================================================================================================================================>


def compute_text_analysis(article_text):
    """
    1> For each of the extracted texts from the article,
       perform textual analysis and compute variables, given in the output structure excel file
       
    2> the output in the exact order as given in the output structure file, “Output Data Structure.xlsx” 
        As follows:
        
        1.POSITIVE SCORE
        2.NEGATIVE SCORE
        3.POLARITY SCORE
        4.SUBJECTIVITY SCORE
        5.AVG SENTENCE LENGTH
        6.PERCENTAGE OF COMPLEX WORDS
        7.FOG INDEX
        8.AVG NUMBER OF WORDS PER SENTENCE
        9.COMPLEX WORD COUNT
        10.WORD COUNT
        11.SYLLABLE PER WORD
        12.PERSONAL PRONOUNS
        13.AVG WORD LENGTH

     
       
    """
    sia = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(article_text)

    word_count = len(nltk.word_tokenize(article_text))
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    avg_words_per_sentence = word_count / len(sentences)
    complex_word_count = len([word for word in nltk.word_tokenize(article_text) if count_syllables(word) >= 3])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_words_per_sentence + percentage_complex_words)
    syllable_per_word = sum(count_syllables(word) for word in nltk.word_tokenize(article_text)) / word_count
    personal_pronouns = len([word for word, pos in nltk.pos_tag(nltk.word_tokenize(article_text)) if pos == "PRP"])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(article_text)) / word_count

    sentiment_scores = sia.polarity_scores(article_text)
    positive_score = sentiment_scores["pos"]
    negative_score = sentiment_scores["neg"]
    polarity_score = sentiment_scores["compound"]
    subjectivity_score = sentiment_scores["compound"]  # Sentiment analysis doesn't provide subjectivity directly

    return positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length,            percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count,            syllable_per_word, personal_pronouns, avg_word_length

if __name__ == "__main__":
    excel_file_path = r"C:\Users\Pruthviraj\Desktop\New folder (4)\input.xlsx"
    df = pd.read_excel(excel_file_path)

    for index, row in df.iterrows():
        url_id = row["URL_ID"]
        url = row["URL"]

        title, article_text = extract_article_text(url)

        if title and article_text:
            save_article_to_folder(url_id, title, article_text)
            print(f"Article {url_id} extracted and saved successfully.")
            
 #<=================================================================================================================================>


            # Compute text analysis variables
            (positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length,
             percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count,
             syllable_per_word, personal_pronouns, avg_word_length) = compute_text_analysis(article_text)
#<=================================================================================================================================>


            # Save the variables to the output structure file
            df.at[index, "POSITIVE SCORE"] = positive_score
            df.at[index, "NEGATIVE SCORE"] = negative_score
            df.at[index, "POLARITY SCORE"] = polarity_score
            df.at[index, "SUBJECTIVITY SCORE"] = subjectivity_score
            df.at[index, "AVG SENTENCE LENGTH"] = avg_sentence_length
            df.at[index, "PERCENTAGE OF COMPLEX WORDS"] = percentage_complex_words
            df.at[index, "FOG INDEX"] = fog_index
            df.at[index, "AVG NUMBER OF WORDS PER SENTENCE"] = avg_words_per_sentence
            df.at[index, "COMPLEX WORD COUNT"] = complex_word_count
            df.at[index, "WORD COUNT"] = word_count
            df.at[index, "SYLLABLE PER WORD"] = syllable_per_word
            df.at[index, "PERSONAL PRONOUNS"] = personal_pronouns
            df.at[index, "AVG WORD LENGTH"] = avg_word_length
        else:
            print(f"Failed to extract the article for {url_id}.")
            
            
#<=================================================================================================================================>

    # Save the updated DataFrame with computed variables to the output file
    output_file_path = r"C:\Users\Pruthviraj\Desktop\New folder (4)\Output Data Structure.xlsx"
    df.to_excel(output_file_path, index=False)
    print("Text analysis completed and output saved successfully.")


# In[ ]:




