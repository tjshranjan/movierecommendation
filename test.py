import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')

def predict_movie(movie):
    df=pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
    # print(df.head())
    movie_titles=pd.read_csv('ml-100k/u.item',sep='\|',encoding='ISO-8859-1',header=None)
    # print(movie_titles.head())
    movie_titles=movie_titles[[0,1]]
    movie_titles.columns=['item_id','title']
    # print(movie_titles.head())
    df=pd.merge(df,movie_titles,on='item_id')
    # print(df.head())
    ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
    ratings['no of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])
    ratings=ratings.sort_values(by='rating')
    # print(ratings)
    # sns.jointplot(x='rating',y='no of ratings',data=ratings)
    # plt.show()
    # print(ratings.sort_values('no of ratings',ascending=False))
    moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
    movie_user_ratings=moviemat[movie]
    # print(movie_user_ratings)
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    # print(similar_to_movie)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    # print(corr_movie.head()) 
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.sort_values('correlation',ascending=False)
    # print(corr_movie)
    corr_movie=corr_movie.join(ratings['no of ratings'])
    corr_movie=corr_movie[corr_movie['no of ratings']>100].sort_values(by='correlation',ascending=False)
    print(corr_movie)

movie_name=input()
print(predict_movie(movie_name))


