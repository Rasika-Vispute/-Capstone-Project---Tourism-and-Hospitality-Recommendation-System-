# Capstone Project - Tourism and Hospitality Recommendation System
# Project Details

## Synopsis:
Developed a machine learning-based recommendation system that suggests travel destinations using content-based and collaborative filtering. Leveraged cosine similarity, Pearson correlation, and KNN to enhance recommendation accuracy and personalize user experiences.

## Key Skills: 
`Machine Learning` · `Recommendation Systems` · `Content-Based Filtering` · `Collaborative Filtering` · `Cosine Similarity` · `Pearson Correlation`· `KNN` 

## Overview:
Tourism can be explained as travel for recreational, leisure or business purposes. The travel and tourism industry are one of the largest and major industries in the world. An important aspect of economic life, the industry is closely associated with a diverse set of sub-major sectors such as travel, food, and accommodation. India as a country has immense potential for travel and tourism.

## Industrial Review:
- On a basic level, the travel or tourism industry is concerned with services for people who have travelled away from their usual place of residence, for a relatively short period. By contrast, the hospitality industry is concerned with services related to leisure and customer satisfaction.
- With a total area of 3,287,263 sq. km extending from the snow-covered Himalayan heights to the tropical rain forests of the south, India has a rich cultural and historical heritage, variety in ecology, terrains, and places of natural beauty spread across the country. This provides a significant opportunity to fully exploit the potential of the tourism sector.
- India is one the most popular travel destinations across the globe has resulted in the Indian tourism and hospitality industry emerging as one of the key drivers of growth in the services sector in India. The tourism Industry in India has significant potential considering that Tourism is an important source of foreign exchange in India like many other countries. The foreign exchange earnings from 2016 to 2019 grew at a CAGR of 7% but dipped in 2020 due to the COVID- 19 pandemic.
- It is widely acknowledged that the tourist and hospitality sector, which encompasses travel and hospitality services like hotels and restaurants, is a development agent, a catalyst for socioeconomic growth, and a significant source of foreign exchange gains in many countries. India's rich and exquisite history, culture, and diversity are showcased through tourism while also providing significant economic benefits. The consistent efforts of the central and state governments have helped the tourism industry to recover from the covid-19 pandemic shock and operate at the Pre pandemic level.

## Business Problem Understanding:
- We are going to recommend the city and places to new people who are planning for tours, based on the existing data. The recommendation system will help and guide a customer to select their preferred city by recommending top 5 cities through content based and collaborative based recommendation systems. Based on the similarity among the user profiles recommendations are made to the target user in collaborative filtering method and based on the similarities among the cities, recommendations are made to the target user in content-based filtering method.
- The details of each user’s preferences, travel history, and search history are stored in the database, and from the dataset, recommendations are made. To provide the recommendation, the system first would have to identify the neighbor user who has a similar preference as the target user. This is done with the help of a neighborhood estimator which uses the user profile data from the user database and the rating data from the rating database.
- Some people want to go for a tour or spend some days outside, but they don’t have good knowledge about the place, or they want only to spend some amount of the wandering outside or on tour, but they don’t know where to spend. 

## Problem Understanding:
One of the major challenges the industry faces is the perception that the country has attained internationally. As one of the most diverse and culturally rich countries, India is the dream destination for avid travelers with a myriad of promising adventures and experiences. Yet, we still only record approximately 10 million visitors annually as compared to a whopping 20 million tourists in Bangkok, Thailand.  As a travel destination, we have lots to offer such as ancient culture, historical heritage, spiritual experiences, beautiful landscapes, natural diversity, adventure, wildlife, and so on. So, keeping all this in mind, in this project we will work on solving the problem of perspective by predicting the type of place the customer wants to go, by doing this we will be able to increase tourism from across the world and in India as well. Increasing growth rate of the Indian tourism sector apart from just temples and cliché, and by this, we will also grab extensive ranges of people from all over the world.

## The current solution to the problem:
**Government-led the most popular and phenomenal** - **Incredible India Campaign** – over the past few years has aided in positioning India as a leading travel destination. However, lack of cohesiveness across various state-led tourism campaigns has limited India as a clichéd, and stereotypical spiritual destination for finding our true selves and attaining Nirvana. We can refer to this tourist section as the Eat-Pray-Love crowd. As a travel destination, we have lots to offer such as ancient culture, historical heritage, spiritual experiences, beautiful landscapes, natural diversity, adventure, wildlife, and so on. However, the tourism growth rate can be tackled by well-planned marketing strategies along with better leisure and tourism atmosphere in terms of safety, security, and quality in accommodation, logistics, and infrastructure. 

## A proposed solution to the problem:
We are recommending the place to the customer based on existing information and an information filtering system that is used to recommend the user’s items based on their previous history or their preferences. So, in our case, we are going to build a model based on a mathematical expression that represents data in the context of a problem, often a business problem. The aim is to go from data to insight. 
In this case, we are going to introduce various types of 

### Recommendation systems: 
1)	Content-based 
2)	Collaborative

## Critical Assessment of Topic Survey:
We collected data for the Tourism sector from different sources. Tourism plays pivotal role in socio-economic development of the country. Travel and Tourism are the largest source of growth in the economy, spreading the cultures all over the world. Tourism industry is facing issues as it could not make visitors as their estimation. 

## Estimations about the tourism: 
India is estimated to contribute 250 billion $ from tourism. 137 million job opportunities in Tourism sector. 56 billion$ in foreign earnings, 25 million foreign arrivals are expected to achieve in next couple of years. In some couple of years, tourism and hospitality is expected to earn $50.9 billion as visitor exports compared with $ 28.9 billion in 2018. International tourist arrivals are expected to reach 30.5 million in years. The tourism sector in India accounted to 39 million jobs, 8 % of total employment of the country. In the next couple of years, it is expected to account for about 53 million jobs. 

## Deep into the tourism growth: 
As the industry is not meeting their expectations, we came across the variable which would contribute towards making industry grow, grab the international tourists all over the world. Recommending the place with ancient culture, historical heritage, spiritual experiences, beautiful landscape, natural diversity, adventure, wildlife and so on. However, the tourism growth rate can be tackled by well-planned marketing strategies along with better leisure and tourism atmosphere in terms of safety, security, and quality in accommodation, logistics, and infrastructure. Make the business multi-seasonal for tourists. Monitor the places for better leisure, experiences, eco-tourism travel would be more safe and secure. Focusing the best places and those which are not in visitor’s lists would help the industry to grow faster. Expanding the vision of industry towards more diverse. Our country can make more experience with ancient cultures of India, wildlife of India, diverse of languages, foods specialty of states, social activities, cultural programs of different states, Sahyadri tour for tourists, river adventure sports, mountain climbing, village lifestyles, historical locations, lifestyle, and culture of seaside strip of India. There are huge number of places to recommend and grab more opportunities in this industry.

## Business understanding:
Recommendation system is defined as an information filtering system that is used to recommend the users items based on their previous history or their preferences. Recommendation System Based on Tourist Attraction which is a location-based Travel recommendation system. To recommend the locations the system uses user based collaborative filtering method. It means that based on the similarity among the user profiles recommendations are made to the target user.

## Data understanding:
We are having the details of customers who went to tour with different perspective. So, in the first data set we have different cities, states and description of those cities. On this dataset we made content-based recommendation system as it needed detailed description of cities to make correct recommendations. In another data set, details of the different cities were given as columns and tourist id as rows. It consists of ratings provided by each tourist id to a particular location. 

## Data preparation:
**1.	Tokenization:** 
Tokenization is the first step in any NLP pipeline. It has an important effect on the rest of your pipeline. A tokenizer breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements. The token occurrences in document can be used directly as a vector representing the document.

**2.	Stop Words:**
A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. We would not want these words to take up space in our database or taking up valuable processing time.

**3.	Stemming:** 
Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers. A stemming algorithm reduces the words “chocolates”, “chocolatey”, “Choco” to the root word, “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve”. Stemming is an important part of the pipe lining process in Natural language processing.

**4.	Lemmatization:**
Lemmatization is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root form, having the same meaning.

## Modelling:
1.	Whenever we apply any algorithm in NLP, it works on numbers. We cannot directly feed our text into that algorithm. Hence, Bag of Words model is used to Pre-process the text by converting it into a bag of words, which keeps a count of the total occurrences of most frequently used words. This model can be visualized using a table, which contains the count of words corresponding to the word itself.
2.	To implement an item based collaborative filtering, KNN is a perfect go-to model and a very good baseline for recommended system development. But what is the KNN? KNN is a non-parametric, lazy learning method. It uses a database in which the data points are separated into several clusters to make inference for new samples.
3.	KNN does not make any assumptions on the underlying data distribution, but it relies on item feature similarity. When KNN makes inference about a city, KNN will calculate the “distance” between the target city and every other city in its database, then it ranks its distances and returns the top K nearest neighbor movies as the most similar city recommendations.

## Conclusion
1. Recommender systems open new opportunities of retrieving personalized information on the Internet. It also helps to alleviate the problem of information overload which is a very common phenomenon with information retrieval systems and enables users to have access to products and services which are not readily available to users on the system. 
2. This projects will help the tourism industry to attract more and more travelers, by recommending accurately the location they want to visit.

##  Contributions
Suggestions, improvements, and issues are welcome. Feel free to fork the repo or raise a pull request!

## Contact
**Rasika Vispute**  
Email: rasikavispute32@gmail.com 
LinkedIn: https://www.linkedin.com/in/rasikavispute/
