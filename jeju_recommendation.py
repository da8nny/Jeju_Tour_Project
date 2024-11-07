import pandas as pd
import streamlit as st
import pickle
import os
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout="wide", initial_sidebar_state="auto", page_title="Jeju", page_icon="🍊")
st.title('제주도 식당 추천 서비스🍊')



path = os.path.dirname(__file__)
with open (path+'/indices_1.pkl', 'rb') as f:
    indices_1 = pickle.load(f)
with open (path+'/cosine_sim_1.pkl', 'rb') as f:
    cosine_sim_1 = pickle.load(f)
with open(path+'/final_downtown_review.pkl', 'rb') as f:
    final_downtown_review = pickle.load(f)
with open(path+'/tfidf_matrix_1.pkl', 'rb') as f:
    tfidf_matrix_1 = pickle.load(f)
with open(path+'/tfidf.pkl', 'rb') as f:
    tfidf= pickle.load(f)

# def get_recommendations(rest, cosine_sim=cosine_sim_1):
#     # 식당명 통해 전체 데이터 기준 그 식당 index값을 얻기
#     idx = indices_1[rest]  # indices_1 필요ss

#     # 코사인 유사도 매트릭스에서 idx에 해당하는 데이터 (idx, 유사도) 형태로 얻기
#     sim_scores = list(enumerate(cosine_sim[idx]))  # cosine_sim_1 필요

#     # 코사인 유사도 기준 내림차순 정렬
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # 자기 자신 제외 5개의 추천 식당 슬라이싱
#     sim_scores = sim_scores[1:6]  # [0:6]이면 본인을 포함하기에 X

#     # 추천 식당명 5개 인덱스 정보 추출
#     rest_indices = [i[0] for i in sim_scores]

#     # 추천 식당과 유사도 반환
#     recommendations = [(final_downtown_review['식당명'].iloc[i], "{:.3f}".format(sim_scores[j][1])) for j, i in enumerate(rest_indices)]
#     # final_downtown_review 필요
 
#     return recommendations

# # 사용자로부터 식당명 입력받기
# # user_input = input("식당명을 입력하세요: ")
# user_input = st.text_input("식당명을 입력하세요: ")
# if user_input:
#     recommendations = get_recommendations(user_input)
#     for rec in recommendations:
#         # print("추천 식당:", rec[0], '&', "유사도:", rec[1])
#         st.write("추천 식당:", rec[0], '&', "유사도:", rec[1])

        
def get_user_input_vector(user_input, tfidf_model):
    return tfidf_model.transform([user_input])

def get_recommendations_by_user_input_with_hotel(user_input, hotel_name, tfidf_model, cosine_sim=cosine_sim_1):
    # 호텔에 부합하는 행들 필터링
    hotel_indices = final_downtown_review[final_downtown_review['숙박업명'] == hotel_name].index

    # Tfidf 백터생성
    user_tfidf_vector = get_user_input_vector(user_input, tfidf_model)

    # 사용자입력 & 호텔 필터링 코사인 유사도 계산
    cosine_sim_user = linear_kernel(user_tfidf_vector, tfidf_matrix_1[hotel_indices])

    # 정렬 (유사도 높은순)
    sim_scores = list(enumerate(cosine_sim_user[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 상위 5개 식당 추출
    sim_scores = sim_scores[:5]
    restaurant_indices = [hotel_indices[i[0]] for i in sim_scores]

    # 추천 식당과 유사도 반환
    recommended_restaurants = final_downtown_review.iloc[restaurant_indices][['식당명', '검색량합계값', '숙박_식당 거리']]
    similarity_scores = [round(i[1], 3) for i in sim_scores]

    return recommended_restaurants, similarity_scores


# 사용자에게 식당 추천하는 함수
def recommend_restaurant():
    #user_hotel = input("어느 호텔에서 묵고 계신가요? ")
    user_hotel = st.text_input("어느 호텔에서 묵고 계신가요? ")
    
    # 입력한 호텔명이 데이터에 있는지 확인
    if user_hotel not in final_downtown_review['숙박업명'].values:
        #print("입력하신 호텔은 존재하지 않습니다.")
        st.write("입력하신 호텔은 존재하지 않습니다.")
        return

    #user_input = input("어떤 식당을 찾으시나요? ")
    user_input = st.text_input("어떤 식당을 찾으시나요? ")

    # 호텔과 사용자 입력에 기반한 식당 추천 및 유사도 가져오기
    recommended_restaurants, similarity_scores = get_recommendations_by_user_input_with_hotel(user_input, user_hotel, tfidf, cosine_sim_1)

    if recommended_restaurants.empty:
        #print("입력하신 조건에 부합하는 식당이 없습니다.")
        st.write("입력하신 조건에 부합하는 식당이 없습니다.")
    if user_hotel and user_input:
        #print("입력하신 조건과 호텔에 부합하는 식당을 아래와 같이 추천드립니다:")
        st.write("입력하신 조건과 호텔에 부합하는 식당을 아래와 같이 추천드립니다:")
        for (restaurant, search_count, distance), score in zip(recommended_restaurants.values, similarity_scores):
            distance = round(distance, 2)
            st.write(f"식당명: {restaurant} / 유사도: {score} / 검색량합계값: {search_count} 건 / 숙박-식당 거리: {distance} km")



def get_recommendations_by_user_input_with_hotel_city(user_input, hotel_name, tfidf_model, cosine_sim=cosine_sim):
    # 호텔에 부합하는 행들 필터링
    hotel_indices_city = final_city_review[final_city_review['숙박업명'] == hotel_name].index

    # TF-IDF 벡터 생성
    user_tfidf_vector_city = get_user_input_vector_city(user_input, tfidf_model)

    # 사용자 입력과 호텔 필터링을 고려한 코사인 유사도 계산
    cosine_sim_user_city = linear_kernel(user_tfidf_vector_city, tfidf_matrix[hotel_indices_city])

    # 유사도가 높은 순으로 정렬
    sim_scores_city = list(enumerate(cosine_sim_user_city[0]))
    sim_scores_city = sorted(sim_scores_city, key=lambda x: x[1], reverse=True)

    # 상위 5개 식당 추출
    sim_scores_city = sim_scores_city[:5]
    restaurant_indices_city = [hotel_indices_city[i[0]] for i in sim_scores_city]

    # 추천 식당과 유사도 반환
    recommended_restaurants_city = final_city_review.iloc[restaurant_indices_city][['식당명', '검색량합계값', '숙박_식당 거리']]
    similarity_scores = [round(i[1], 3) for i in sim_scores_city]

    return recommended_restaurants_city, similarity_scores