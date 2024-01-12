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
from textblob import Word



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
    
    return results

def clean_str(string):
    '''
    tokenization, cleaning for dataset
    '''
    string=str(string)
    string = re.sub(r"\'s", "", string)

    string = re.sub(r"\'ve", "", string)

    string = re.sub(r"\'t", "", string) 
    
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'lls", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() 
def predict():
    features = clean_str(content)
   
    
    print("features type",type(features)," and value -",features)
    features = features.split(" ")
    transformed_data= tfidf.transform(features)
    print("\n\n\n\n prediction \n\n\n\n",model.predict(transformed_data))
    print("\n\n\n\n features \n\n\n\n",features)
    predicted = model.predict(transformed_data)[0]
    # return features
    # for i ,value in enumerate(features):
    #     #features[i]=pre_process(value)
    #     features[i]= (value)
    # return features[i]
    
    cv = CountVectorizer(stop_words='english')
    word_count_vector=cv.fit_transform(features)
    


    
    doc=features[0]
    #Generate tf-idf for the given document
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    tfidf_vector =tfidf_transformer.transform(cv.transform([doc]))
    #Sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf_vector.tocoo())
    #Only needed once 
    #msg="Put some more important words and try for recommendation"
    feature_names = cv.get_feature_names_out()


    #Extract only the top n: n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,3)
    # return keywords
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
    # print("Length of :",response_json['totalResults'])
    if 'totalResults' not in response_json:
        return [["The response is empty"]],[["https://images.app.goo.gl/npe1pLsjBr8u1V9w5"]]
        
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
    #print('writing csv file...')
    df.to_csv('./current.csv', index = False)

    cur_csv=pd.read_csv("./current.csv")
    
    
    text_features = tfidf.transform(features)
    prediction = model.predict(text_features)

    # move articles to an array
    articles = cur_csv.description.values

    # move article web_urls to an array
    web_url = cur_csv.url.values
    img_url =cur_csv.image_url.values
    title =cur_csv.title.values

        # shuffle these two arrays 
    articles, web_url,img_url,title = shuffle(articles,web_url, img_url,title,random_state=4)
        
    n=5
    xtrain = articles[:]
    xtrain_urls = web_url[:]
    x_img_url=img_url[:]
    #tfidf.fit(xtrain)
    x_tfidf = tfidf.transform(xtrain)

    top_recommendations, recommended_urls, sorted_sim_scores,rec_img_url = top_articles(x_tfidf, xtrain, text_features,xtrain_urls,x_img_url,title,5)
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
st.sidebar.header("Write notes below")
content = st.sidebar.text_area("Enter your notes here")


# Button to trigger predictions
if st.sidebar.button("Predict"):
    # Your prediction logic here
    # ...
    prediction_arr,recommended_urls,predicted = predict()
    # keywords=predict()
    # st.write("Prediction:", keywords)
    
    print("prediction array :\n\n",prediction_arr)
    print("recommended_urls array :\n\n",recommended_urls)
    
    for prediction_list, url_list in zip(prediction_arr, recommended_urls):
        # Extract the first prediction from the list (if available)
        prediction = prediction_list[0] if prediction_list.size > 0 else None
        # Extract the first URL from the array (if available)
        url = url_list[0] if url_list.size > 0 else None

        if prediction and url:
            st.write(f"{prediction}")
            st.markdown(f"[Recommended URL]({url})")
            st.write(f"Predicted under {predicted}")
            st.write("---")  # Add a separator between each pair
# Streamlit app footer
st.sidebar.markdown("---")
st.sidebar.markdown("Your additional information or links go here.")


