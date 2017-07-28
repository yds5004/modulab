import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv 전체 title
Data_Title=['color', 'director_name', 'num_critic_for_reviews', 'duration',
            'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name','actor_1_facebook_likes',
            'gross','genres', 'actor_1_name', 'movie_title', 'num_voted_users','cast_total_facebook_likes',
            'actor_3_name','facenumber_in_poster', 'plot_keywords', 'movie_imdb_link','num_user_for_reviews',
            'language','country', 'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',
            'imdb_score	aspect_ratio','movie_facebook_likes']

# 사용할 title
Select_Title=['num_critic_for_reviews', 'duration','director_facebook_likes','actor_3_facebook_likes',
              'actor_1_facebook_likes', 'gross','num_voted_users','cast_total_facebook_likes',
              'facenumber_in_poster','num_user_for_reviews','content_rating','budget','actor_2_facebook_likes','imdb_score',
              'aspect_ratio','movie_facebook_likes']

# csv 읽기
Movie_Data = pd.read_csv('resources/movie_metadata.csv')
# content rating의 값에 대해서 set
index = list(set(Movie_Data['content_rating'].values.tolist()))

# content rating의 set 값을 key로, 0부터 index를 값으로 만듬
content_rating_index = {}
for i, idx in enumerate(index):
    content_rating_index[idx] = i

def setIdToValue(value):
    ret = content_rating_index[value]
    return ret

Movie_Data['content_rating']=Movie_Data['content_rating'].apply(setIdToValue)
Sel_MovieData=Movie_Data[Select_Title]

Movie_Data.head(1)


Sel_MovieData.head(1)


Positive_Th=5

Pos_Data=Movie_Data[Sel_MovieData['imdb_score']>Positive_Th]
Neg_Data=Movie_Data[Sel_MovieData['imdb_score']<=Positive_Th]

num_item=len(Select_Title)
print(num_item)


fig=plt.figure(figsize=(20,20))

fig.add_subplot(2,2,1)
Pos_Data[Select_Title[0]].plot(kind='kde')
Neg_Data[Select_Title[0]].plot(kind='kde')
plt.legend(['P','N'])
plt.title(Select_Title[0])



fig=plt.figure(figsize=(20,20))
num_col=2
num_row= num_item/num_col
if num_item%num_col:
    num_row+=1

for i in range(num_item):
    fig.add_subplot(num_row,num_col,i+1)
    Pos_Data[Select_Title[i]].plot(kind='kde')
    Neg_Data[Select_Title[i]].plot(kind='kde')
    plt.legend(['P','N'])
    plt.title(Select_Title[i])


plt.figure(figsize=(2,2))

def sigmoid(t):
    return (1/(1+np.e**(-t)))

plot_range=np.arange(-6,6,0.1)

y=sigmoid(plot_range)
plt.plot(plot_range, y, color="red")