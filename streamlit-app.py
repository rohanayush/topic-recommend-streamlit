import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle
import requests
from textblob import Word
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
# Download English stopwords
nltk.download('stopwords')

# Download Japanese stopwords
nltk.download('stopwords-jp')


# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('trained_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


#n = 5 is default values for number of similar articles to be found, if n= 7 is passed then 7 similar articles would be found
def top_articles(xtrain_tfidf, xtrain, texts,xtrain_urls,x_img_url,title,n=10):
    
    similarity_scores = xtrain_tfidf.dot(texts.toarray().T)
    #Calculating similarity between notes and articles and scores are stored
    
    sorted_indices = np.argsort(similarity_scores, axis = 0)[::-1]
    
    sorted_scores =similarity_scores[sorted_indices]
    
    #get n topmost similar articles
    top_rec = xtrain[sorted_indices[:n]]
    
    #get top n corresponding urls
    rec_urls = xtrain_urls[sorted_indices[:n]]
    rec_img_urls = x_img_url[sorted_indices[:n]]
    rec_title = title[sorted_indices[:n]]
    return top_rec, rec_urls,sorted_scores,rec_img_urls


def sort_coo(coo_matrix):
    tuples= zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x:(x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    print("feature_names",feature_names)
    sorted_items = sorted_items[:topn]
    
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
        
    # create a dictionary of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    print("Getting results",results)
    
    return results

def clean_str(string):
    '''
    Tokenization, cleaning, stopword removal, and lemmatization for the dataset
    '''
    string = str(string)
    
    # Remove unnecessary characters
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    
    # Tokenization
    tokens = string.strip().lower().split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    cleaned_string = " ".join(tokens)
    
    return cleaned_string

def predict(max):
    features = clean_str(content)
    print("Full feauture", features)
    features = features.split(" ")
    print("Full feauture array", features)
    
    transformed_data= tfidf.transform(features)
    predicted = model.predict(transformed_data)[0]
    # return features
    # for i ,value in enumerate(features):
    #     #features[i]=pre_process(value)
    #     features[i]= (value)
    # return features[i]
    
    cv = CountVectorizer(stop_words='english')
    word_count_vector=cv.fit_transform(features)
    print("word_count_vector",word_count_vector)
    
    doc=features[0]
    print("doc",doc)
    #Generate tf-idf for the given document
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    tfidf_vector =tfidf_transformer.transform(cv.transform([doc]))
    #Sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf_vector.tocoo())
    
    feature_names = cv.get_feature_names_out()
    # feature_names = feature_names[:10]
    print("feature names",feature_names)


    #Extract only the top n
    keywords=extract_topn_from_vector(feature_names,sorted_items,3)
    # return keywords
    print("\n\n\n%%%%%%%%%%%%\n\n",keywords)
    qu=""
    for k in keywords:
        qu+=qu+k+" "
    cn='in'
    secret='52c11279b478428daeaf4bfecc1c684a'
    # Define the endpoint
    url = 'https://newsapi.org/v2/everything?'
    # Specify the query and number of returns
    parameters = {
    'q': qu, # query phrase
    'pageSize': 100,
      # maximum is 100
    'apiKey': secret # your own API key
    
    }
    # Make the request
    response = requests.get(url, params=parameters)
    # Convert the response to JSON format and pretty print it
    response_json = response.json()
    # print("\n\n\n\n Response json \n\n\n\n",response_json)
    # print("Length of :",response_json['totalResults'])
    if 'totalResults' not in response_json:
        return [["The response is empty"]],[["https://images.app.goo.gl/npe1pLsjBr8u1V9w5"]],"Empty"
        
    news=response_json
    url_col =[]
    img=[]
    title=[]
    description=[]

    for y in news.keys():
        if (y =='articles'):
            art=news[y]
            for i in range(len(art)):
                if(art[i]['description'] != None):                    
                    description.append(art[i]['description']+ art[i]['title'])
                else:
                    description.append(art[i]['description'])
                url_col.append(art[i]['url'])
                img.append(art[i]['urlToImage'])
                title.append(art[i]['title'])
                    
			    
    for index, value in enumerate(description):
    
        description[index] = ' '.join([Word(word) for word in clean_str(value).split()])



    for index, value in enumerate(title):
        title[index] = ' '.join([Word(word) for word in clean_str(value).split()])	
    dataframe = {'title':title,'url':url_col,'description':description ,'image_url':img}
    df = pd.DataFrame(dataframe)
    
    
    text_features = tfidf.transform(features)
    prediction = model.predict(text_features)

    # move articles to an array
    articles = df.description.values

    # move article web_urls to an array
    web_url = df.url.values
    img_url =df.image_url.values
    title =df.title.values

        # shuffle these two arrays 
    articles, web_url,img_url,title = shuffle(articles,web_url, img_url,title,random_state=4)
        
    n=max
    xtrain = articles[:]
    xtrain_urls = web_url[:]
    x_img_url=img_url[:]
    #tfidf.fit(xtrain)
    x_tfidf = tfidf.transform(xtrain)

    top_recommendations, recommended_urls, sorted_sim_scores,rec_img_url = top_articles(x_tfidf, xtrain, text_features,xtrain_urls,x_img_url,title,n)
    # st.write("from dunction"+prediction)
    print(top_recommendations)
    print(recommended_urls)
    print(predicted)
    return top_recommendations,recommended_urls,predicted




app_title = "News Recommendations"
st.set_page_config(page_title=app_title, page_icon=":smiley:")

# Streamlit app layout
st.title(app_title)

# Sidebar with input form
# st.header("Write notes below")
content = st.text_area("Enter your notes here")
selected_value = st.slider('Select max number of results that can be fetched', 1, 10, 1)


# Button to trigger predictions
if st.button("Recommend news"):
    prediction_arr,recommended_urls,predicted = predict(selected_value)
    st.header("Top recommendations:")
    st.write("---")  # Add a separator between each pair
    
    print(prediction_arr,recommended_urls)
        # st.write(predicted)
    if(predicted !="Empty"):
        for prediction_list, url_list in zip(prediction_arr, recommended_urls):
        # Extract the first prediction from the list (if available)
            prediction = prediction_list[0] if prediction_list.size > 0 else None
            # Extract the first URL from the array (if available)
            url = url_list[0] if url_list.size > 0 else None
            if prediction and url:
               
               st.write(f"{prediction}")
               st.markdown(f"[Recommended URL]({url})")
               st.write("---")  # Add a separator between each pair
               
            else:
                st.write("There are infinity to be explored! Try something new")
            # st.write(f"Predicted under {predicted}")
            # st.write("---")  # Add a separator between each pair
    else:
        st.write("There are infinity to be explored! Try something new")
        
# Streamlit app footer
name = "Rohan Ayush"



# Twitter link with emoji
twitter_link = "[Twitter](https://twitter.com/rohanayush) :smiley:"

# LinkedIn link with emoji
linkedin_link = "[LinkedIn](https://www.linkedin.com/in/rohanayush) :rocket:"

# Display the information using markdown
st.markdown(f"Created by {name}")
st.markdown(twitter_link)
st.markdown(linkedin_link)
st.markdown("---")

# st.write("Created by Rohan Ayush")



