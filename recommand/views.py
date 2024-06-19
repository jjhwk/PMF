
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from tensorflow.python.keras.models import load_model
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.vgg19 import preprocess_input
from keras.api.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from rembg import remove
from PIL import Image
import numpy as np
import tensorflow as tf
from django.http import HttpResponse
import os
import random
from scipy.spatial.distance import cosine
import concurrent.futures

import sqlite3
import requests
from io import BytesIO
from django.core.paginator import Paginator

# Text
from konlpy.tag import Okt
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import torch
import json
import re
from nltk.tokenize import word_tokenize



# Create your views here.

def main(request):
    return render(request, "main.html")

def woman(request):
    return render(request, "woman/woman.html")


# man 유사이미지 불러오기
# 모델 불러오기
model_path_clothes = r'C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\model\\best_VGG19.keras'
model_clothes = tf.keras.models.load_model(model_path_clothes)

model_path_style = r"C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\model\\best_model_v4.keras"
model_style = tf.keras.models.load_model(model_path_style)

model_path_style_w = r'C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\model\\best_women_model_v2.keras'
model_style_w = tf.keras.models.load_model(model_path_style_w)

# 남자 텍스트 모델
m_text_model_path = r"C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\텍스트모델\\mpnet_model"
m_text_model = SentenceTransformer(m_text_model_path)

# 남자 임베딩
m_embeddings = torch.load(r"C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\텍스트모델\\embeddings\\mpnet_embeddings.pt")


# 여자 텍스트 모델
w_text_model_path = r"C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\텍스트모델\\woman_mpnet_model"
w_text_model = SentenceTransformer(w_text_model_path)

# 여자 임베딩
w_embeddings = torch.load(r"C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\텍스트모델\\embeddings\\woman_mpnet_embeddings.pt")

class ImageSimilarityCalculator:
    def __init__(self):
        self.img2_path_list = []
        self.similarity_list = []
        self.best_index_list = []
        self.img2_best_path_list = []
        self.base_model = None  
        self.features1 = None  # 이미지 1의 특징을 저장할 변수
    
    # 각각의 이미지 경로 가져오는 함수
    def get_image_path(self, image_path, style, img1_path, num=150): 
        path = os.path.join(image_path, style)                        
        img_list = os.listdir(path)                          # 원본 이미지 리스트
        random_img_list = random.sample(img_list, num)       # 랜덤하게 추출한 이미지 리스트
        
        # 특정 style 이미지 경로를 가져와 img2_path_list 에 추가
        for img_name in random_img_list: 
            img2_path = os.path.join(path, img_name)        # 랜덤하게 추출한 이미지 리스트의 이미지 경로 = img2_path
            self.img2_path_list.append(img2_path)           # img2_path 를 img2_path_list에 추가
            
        # 이미지 1의 특징 추출
        self.features1 = self.extract_features(img1_path)
            
        return self.calculate_similarity()
    
    # 이미지 유사도 파악하는 함수
    def calculate_similarity(self):
        # 이미지 유사성 파악을 병렬로 진행
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [] # 등록 한 이미지와 무작위로 선정된 이미지 의 유사성 저장되는 리스트
            
            # 병렬로 진행하면 입력되는 순서대로 저장 되지 않기 때문에 인덱스와 같이 저장
            for index, img2_path in enumerate(self.img2_path_list):
                futures.append(executor.submit(self.image_similarity, index, img2_path))
            for future in concurrent.futures.as_completed(futures):
                self.similarity_list.append(future.result())                

        return self.top_10()

    def top_10(self):        
        # x[1] : 유사도 기준으로 내림 차순으로 정렬
        sorted_similarity_with_index = sorted(self.similarity_list, key=lambda x: x[1], reverse=True) 
                
        # 10개 추려서 top_10_with_index에 저장
        top_10_with_index = sorted_similarity_with_index[:10]
        
        # top_10_with_index에서 index 값들만 추려서 top_10_indices에 저장
        top_10_indices = [index for index, _ in top_10_with_index]
        
        self.best_index_list = top_10_indices
                
        return self.top10_path()
    

    # top10의 이미지 주소
    def top10_path(self):
        for best_index in self.best_index_list:
            img_path = self.img2_path_list[best_index]
            self.img2_best_path_list.append(img_path) 

        return self.show_images()

    # 이미지 전처리
    def preprocess_image(self, img_path):
        # PIL 이미지 로드 후 크기 조정
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess_input(img_array)
        return img_array
    
    # 모델 초기화
    def init_model(self):
        if self.base_model is None:
            self.base_model = ResNet50(weights='imagenet', include_top=False)
    
    # 이미지 특징 추출
    def extract_features(self, img_path):
        self.init_model()  # 모델 초기화
        preprocessed_img = self.preprocess_image(img_path)
        features = self.base_model.predict(preprocessed_img)
        return features.flatten()
    
    # 이미지간 유사성 계산
    def image_similarity(self, index, img2_path):       
        features2 = self.extract_features(img2_path)
        similarity = 1 - cosine(self.features1, features2)
        return index, similarity

    # 이미지 출력
    
    # ImageSimilarityCalculator 클래스 내에 show_images 메서드를 추가합니다.
    def show_images(self):
        db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'

        # SQLite 데이터베이스에 연결
        conn = sqlite3.connect(db_path)

        img_id_list = []
        for image_path in self.img2_best_path_list:
            # 이미지 id 값 가져오기
            img_id = (image_path.split("_")[-1]).split(".")[0]
            img_id_list.append(img_id)

        # 튜플로 변환
        img_id_tuple = tuple(img_id_list)

        # 쿼리 작성
        query = f"SELECT rowid, url FROM men_style_all WHERE style_id IN {img_id_tuple}"

        # 쿼리 실행 및 결과 가져오기
        cursor = conn.cursor()
        cursor.execute(query)

        img_url_list = []
        # 결과 처리
        for row in cursor.fetchall():
            row_id, url = row
            img_url_list.append(url)
        # 연결 종료
        conn.close()
        return img_url_list
    


class ImageSimilarityCalculator_w:
    def __init__(self):
        self.img2_path_list = []
        self.similarity_list = []
        self.best_index_list = []
        self.img2_best_path_list = []
        self.base_model = None  
        self.features1 = None  # 이미지 1의 특징을 저장할 변수
    
    # 각각의 이미지 경로 가져오는 함수
    def get_image_path(self, image_path, style, img1_path, num=150): 
        path = os.path.join(image_path, style)                        
        img_list = os.listdir(path)                          # 원본 이미지 리스트
        random_img_list = random.sample(img_list, num)       # 랜덤하게 추출한 이미지 리스트
        
        # 특정 style 이미지 경로를 가져와 img2_path_list 에 추가
        for img_name in random_img_list: 
            img2_path = os.path.join(path, img_name)        # 랜덤하게 추출한 이미지 리스트의 이미지 경로 = img2_path
            self.img2_path_list.append(img2_path)           # img2_path 를 img2_path_list에 추가
            
        # 이미지 1의 특징 추출
        self.features1 = self.extract_features(img1_path)
            
        return self.calculate_similarity()
    
    # 이미지 유사도 파악하는 함수
    def calculate_similarity(self):
        # 이미지 유사성 파악을 병렬로 진행
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [] # 등록 한 이미지와 무작위로 선정된 이미지 의 유사성 저장되는 리스트
            
            # 병렬로 진행하면 입력되는 순서대로 저장 되지 않기 때문에 인덱스와 같이 저장
            for index, img2_path in enumerate(self.img2_path_list):
                futures.append(executor.submit(self.image_similarity, index, img2_path))
            for future in concurrent.futures.as_completed(futures):
                self.similarity_list.append(future.result())                

        return self.top_10()

    def top_10(self):        
        # x[1] : 유사도 기준으로 내림 차순으로 정렬
        sorted_similarity_with_index = sorted(self.similarity_list, key=lambda x: x[1], reverse=True) 
                
        # 10개 추려서 top_10_with_index에 저장
        top_10_with_index = sorted_similarity_with_index[:10]
        
        # top_10_with_index에서 index 값들만 추려서 top_10_indices에 저장
        top_10_indices = [index for index, _ in top_10_with_index]
        
        self.best_index_list = top_10_indices
                
        return self.top10_path()
    

    # top10의 이미지 주소
    def top10_path(self):
        for best_index in self.best_index_list:
            img_path = self.img2_path_list[best_index]
            self.img2_best_path_list.append(img_path) 

        return self.show_images()

    # 이미지 전처리
    def preprocess_image(self, img_path):
        # PIL 이미지 로드 후 크기 조정
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess_input(img_array)
        return img_array
    
    # 모델 초기화
    def init_model(self):
        if self.base_model is None:
            self.base_model = ResNet50(weights='imagenet', include_top=False)
    
    # 이미지 특징 추출
    def extract_features(self, img_path):
        self.init_model()  # 모델 초기화
        preprocessed_img = self.preprocess_image(img_path)
        features = self.base_model.predict(preprocessed_img)
        return features.flatten()
    
    # 이미지간 유사성 계산
    def image_similarity(self, index, img2_path):       
        features2 = self.extract_features(img2_path)
        similarity = 1 - cosine(self.features1, features2)
        return index, similarity

    # 이미지 출력
    
    # ImageSimilarityCalculator 클래스 내에 show_images 메서드를 추가합니다.
    def show_images(self):
        db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'

        # SQLite 데이터베이스에 연결
        conn = sqlite3.connect(db_path)

        img_id_list = []
        for image_path in self.img2_best_path_list:
            # 이미지 id 값 가져오기
            img_id = (image_path.split("_")[-1]).split(".")[0]
            img_id_list.append(img_id)

        # 튜플로 변환
        img_id_tuple = tuple(img_id_list)

        # 쿼리 작성
        query = f"SELECT rowid, url FROM women_style_all WHERE style_id IN {img_id_tuple}"

        # 쿼리 실행 및 결과 가져오기
        cursor = conn.cursor()
        cursor.execute(query)

        img_url_list = []
        # 결과 처리
        for row in cursor.fetchall():
            row_id, url = row
            img_url_list.append(url)
        # 연결 종료
        conn.close()
        return img_url_list
    
        
def man(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            try:
                file = request.FILES['image']
                
                # Save the uploaded file
                file_name = default_storage.save(file.name, ContentFile(file.read()))
                file_url = default_storage.url(file_name)

                # Process the image
                image = Image.open(file)
                # 배경제거
                image = remove(image)  

                # 4차원으로 변경
                image = image.convert("RGBA")
                # 4차원의 흰색 배경 생성
                background = Image.new("RGBA", image.size, (0, 0, 0))
                # 4차원으로 변경된 이미지와 흰색 배경 합성
                image1 = Image.alpha_composite(background, image)
                # 다시 3차원 변환
                image1 = image1.convert("RGB")
                # 224, 224로 높이 너비 변경 
                image1 = image1.resize((224, 224))               
                input_data = img_to_array(image1)
                input_data = np.expand_dims(input_data, axis=0)
                input_data = preprocess_input(input_data)
                input_data = input_data / 255.0

                # Predict if it's clothes
                prediction = model_clothes.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)[0]
                styles = ['clothes', 'not_clothes']  # Ensure this matches your model's classes
                predicted_style = styles[predicted_class]

                if predicted_style == 'clothes':
                    # 배경을 검정색으로 설정
                    background = Image.new("RGBA", image.size, (0, 0, 0))
                    image = Image.alpha_composite(background, image)
                    image = image.convert("RGB")
                    image = image.resize((224, 224))

                    img_array = np.asarray(image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  # 이미지를 0과 1 사이로 정규화

                    # Predict style
                    predictions = model_style.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    class_labels = ['american', 'casual', 'chic', 'dandy', 'formal', 'gorpcore', 'sports', 'street']
                    predicted_class_label = class_labels[predicted_class_index]

                    # Find similar images
                    img1_path = os.path.join(settings.MEDIA_ROOT, file_name)
                    image_similarity_calculator = ImageSimilarityCalculator()
                    similar_images = image_similarity_calculator.get_image_path(r'C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\media\\media\\image_nuggi\\men\\', predicted_class_label, img1_path)
                    
                    random_similar_image = random.choice(similar_images) if similar_images else None

                    return render(request, 'man/man.html', {
                        'prediction': predicted_class_label, 
                        'image_url': file_url,
                        'img_url_list': similar_images,
                        'random_similar_image': random_similar_image,
                    })
                else:
                    return render(request, 'man/man.html', {'prediction': 'WRONG IMAGE', 'image_url': file_url})

            except Exception as e:
                print(f"Error: {e}")  # Log the error
                return render(request, 'man/man.html', {'error': str(e)})
        else:
            return render(request, 'man/man.html', {'error': 'No image uploaded'})
    return render(request, 'man/man.html')

def woman(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            try:
                file = request.FILES['image']
                
                # Save the uploaded file
                file_name = default_storage.save(file.name, ContentFile(file.read()))
                file_url = default_storage.url(file_name)

                # Process the image
                image = Image.open(file)
                # Background removal
                
                 # 배경제거
                image = remove(image)  

                # 배경을 검정색으로 설정
                image = image.convert("RGBA")
                background = Image.new("RGBA", image.size, (255, 255, 255))
                image1 = Image.alpha_composite(background, image)
                image1 = image1.convert("RGB")
                image1 = image1.resize((224, 224))

                input_data = img_to_array(image1)                
                input_data = np.expand_dims(input_data, axis=0)
                input_data = preprocess_input(input_data)
                input_data = input_data / 255.0

                # Predict if it's clothes
                prediction = model_clothes.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)[0]
                styles = ['clothes', 'not_clothes']  # Ensure this matches your model's classes
                predicted_style = styles[predicted_class]

                if predicted_style == 'clothes':
                    # Further classify the clothing style
                    background = Image.new("RGBA", image.size, (0, 0, 0))
                    image = Image.alpha_composite(background, image)
                    image = image.convert("RGB")
                    image = image.resize((224, 224))

                    img_array = np.asarray(image)                    
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  # 이미지를 0과 1 사이로 정규화

                    # Predict style
                    predictions = model_style_w.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    class_labels = ['casual', 'chic', 'formal', 'girlish', 'romantic', 'sports', 'street']
                    predicted_class_label = class_labels[predicted_class_index]

                    # Find similar images
                    img1_path = os.path.join(settings.MEDIA_ROOT, file_name)
                    image_similarity_calculator = ImageSimilarityCalculator_w()
                    similar_images = image_similarity_calculator.get_image_path(r'C:\\Users\\ITSC\\Desktop\\final_240607\\PMF_DB\\media\\media\\image_nuggi\\women\\', predicted_class_label, img1_path)
                    
                    return render(request, 'woman/woman.html', {
                        'prediction': predicted_class_label, 
                        'image_url': file_url,
                        'img_url_list': similar_images
                    })
                else:
                    return render(request, 'woman/woman.html', {'prediction': 'WRONG IMAGE', 'image_url': file_url})

            except Exception as e:
                print(f"Error: {e}")  # Log the error
                return render(request, 'woman/woman.html', {'error': str(e)})
        else:
            return render(request, 'woman/woman.html', {'error': 'No image uploaded'})
    return render(request, 'woman/woman.html')



def img_detail(request, img_url):
    # 데이터베이스 연결
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # men_style_all 테이블에서 clothe1 ~ clothe6 값을 가져오기
        query_clothes = "SELECT clothe1, clothe2, clothe3, clothe4, clothe5, clothe6 FROM men_style_all WHERE url = ?"
        cursor.execute(query_clothes, (img_url,))
        clothes_row = cursor.fetchone()
        if clothes_row is not None:
            clothes_ids = [clothes_row[i] for i in range(6) if clothes_row[i] is not None]
        else:
            clothes_ids = []

        # men_clothes_info 테이블에서 해당 clothe id에 해당하는 img_url_id를 가져오기
        related_img_urls = []
        if clothes_ids:
            query_related_imgs = "SELECT img_url_id FROM men_clothes_info WHERE id IN ({seq})".format(
                seq=','.join(['?']*len(clothes_ids)))
            cursor.execute(query_related_imgs, clothes_ids)
            related_img_rows = cursor.fetchall()
            for row in related_img_rows:
                related_img_urls.append(row[0])

        # 디버깅: related_img_urls 출력
        print("related_img_urls:", related_img_urls)

    finally:
        # 데이터베이스 연결 종료
        cursor.close()
        conn.close()

    if related_img_urls:
        # img_url과 관련된 이미지들의 URL을 템플릿으로 전달
        return render(request, 'img_detail.html', {'img_id': img_url, 'related_img_urls': related_img_urls})
    else:
        # 이미지를 찾을 수 없음을 알리는 페이지 반환
        return render(request, 'img_detail.html', {'img_id': img_url})


def img_detail_w(request, img_url):
    # 데이터베이스 연결
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # men_style_all 테이블에서 clothe1 ~ clothe6 값을 가져오기
        query_clothes = "SELECT clothe1, clothe2, clothe3, clothe4, clothe5, clothe6 FROM women_style_all WHERE url = ?"
        cursor.execute(query_clothes, (img_url,))
        clothes_row = cursor.fetchone()
        if clothes_row is not None:
            clothes_ids = [clothes_row[i] for i in range(6) if clothes_row[i] is not None]
        else:
            clothes_ids = []

        # men_clothes_info 테이블에서 해당 clothe id에 해당하는 img_url_id를 가져오기
        related_img_urls = []
        if clothes_ids:
            query_related_imgs = "SELECT img_url_id FROM women_clothes_info WHERE id IN ({seq})".format(
                seq=','.join(['?']*len(clothes_ids)))
            cursor.execute(query_related_imgs, clothes_ids)
            related_img_rows = cursor.fetchall()
            for row in related_img_rows:
                related_img_urls.append(row[0])

        # 디버깅: related_img_urls 출력
        print("related_img_urls:", related_img_urls)

    finally:
        # 데이터베이스 연결 종료
        cursor.close()
        conn.close()

    if related_img_urls:
        # img_url과 관련된 이미지들의 URL을 템플릿으로 전달
        return render(request, 'img_detail_w.html', {'img_id': img_url, 'related_img_urls': related_img_urls})
    else:
        # 이미지를 찾을 수 없음을 알리는 페이지 반환
        return render(request, 'img_detail_w.html', {'img_id': img_url})


# Text 함수

okt = Okt()
# 영어 불용어 리스트
stop_words = set(stopwords.words('english'))

# 전처리 함수
def preprocess(text):
    # 소문자 변환
    text = text.lower()
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    
    # 형태소 분석과 영문 토크나이저를 함께 사용
    tokens = []
    for word in text.split():
        if re.match(r'[a-zA-Z]+', word):
            # 영어 단어 토크나이징
            word_tokens = word_tokenize(word)
            # 영어 불용어 제거 후 tokens에 
            tokens.extend([w for w in word_tokens if not w in stop_words])
        else:
            # 한국어 형태소 분석
            tokens.extend(okt.morphs(word))
    
    return ' '.join(tokens)

# main, sub 일치하는 데이터 가져오는 함수
def filter_data(rows, col_names, main, sub):
    filtered_data = []
    main_idx = col_names.index('main')
    sub_idx = col_names.index('sub')

    for row in rows:
        if row[main_idx] == main and row[sub_idx] == sub:
            filtered_data.append(row)

    return filtered_data

def find_weighted_terms(title, synonyms):
    weighted_terms = {}
    found_terms = []
    
    for key, values in synonyms.items():
        for value in values:
            if value in title:
                weighted_terms[key] = 2.0  # 기본 가중치 값을 2.0로 설정
                found_terms.append(value)
                break
                
    return weighted_terms, found_terms


# 남자 텍스트 모델
def m_filter_mpnet_cos_items(rows, col_names, main, sub, title, top_k = 5, synonyms = None):
    filtered_data = filter_data(rows, col_names, main, sub)
    db_title = [row[col_names.index('title')] for row in filtered_data]
    db_urls = [row[col_names.index('img_url_id')] for row in filtered_data]
    db_ids = [row[col_names.index('id')] for row in filtered_data]
    
    # 필터링된 임베딩 추출
    filtered_embeddings = m_embeddings[[rows.index(row) for row in filtered_data]]
    
    # 입력된 title 전처리 및 임베딩 생성
    title_processed = preprocess(title)
    title_embedding = m_text_model.encode([title_processed], convert_to_tensor = True)
    
    # 특정 단어에 가중치 부여
    weighted_terms, found_terms = [], []
    if synonyms:
        weighted_terms, found_terms = find_weighted_terms(title_processed, synonyms)
        for term, weight in weighted_terms.items():
            term_embedding = m_text_model.encode([term], convert_to_tensor = True)
            title_embedding += term_embedding * weight
    
    # 코사인 유사도 계산
    cosine_scores = util.pytorch_cos_sim(title_embedding, filtered_embeddings)[0]
    
    # 유사도 점수가 높은 상위 top_k 인덱스 찾기
    top_k_indices = torch.topk(cosine_scores, k = len(cosine_scores)).indices.numpy()
    
    # 상위 결과에서 사용자 입력 텍스트와 동일한 텍스트 제외하고 top_k 개수만큼 선택
    results = []
    count = 0
    for idx in top_k_indices:
        if db_title[idx] != title:
            results.append((db_title[idx], db_urls[idx], db_ids[idx]))
            count += 1
        if count == top_k:
            break
    
    # 가중치가 부여된 단어들 반환
    return results

# 여자 텍스트 모델
def w_filter_mpnet_cos_items(rows, col_names, main, sub, title, top_k = 5, synonyms = None):
    filtered_data = filter_data(rows, col_names, main, sub)
    db_title = [row[col_names.index('title')] for row in filtered_data]
    db_urls = [row[col_names.index('img_url_id')] for row in filtered_data]
    db_ids = [row[col_names.index('id')] for row in filtered_data]
    
    # 필터링된 임베딩 추출
    filtered_embeddings = w_embeddings[[rows.index(row) for row in filtered_data]]
    
    # 입력된 title 전처리 및 임베딩 생성
    title_processed = preprocess(title)
    title_embedding = w_text_model.encode([title_processed], convert_to_tensor = True)
    
    # 특정 단어에 가중치 부여
    weighted_terms, found_terms = [], []
    if synonyms:
        weighted_terms, found_terms = find_weighted_terms(title_processed, synonyms)
        for term, weight in weighted_terms.items():
            term_embedding = w_text_model.encode([term], convert_to_tensor = True)
            title_embedding += term_embedding * weight
    
    # 코사인 유사도 계산
    cosine_scores = util.pytorch_cos_sim(title_embedding, filtered_embeddings)[0]
    
    # 유사도 점수가 높은 상위 top_k 인덱스 찾기
    top_k_indices = torch.topk(cosine_scores, k = len(cosine_scores)).indices.numpy()
    
    # 상위 결과에서 사용자 입력 텍스트와 동일한 텍스트 제외하고 top_k 개수만큼 선택
    results = []
    count = 0
    for idx in top_k_indices:
        if db_title[idx] != title:
            results.append((db_title[idx], db_urls[idx], db_ids[idx]))
            count += 1
        if count == top_k:
            break
    
    # 가중치가 부여된 단어들 반환
    return results


def clothe_info(request, related_img_url):
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
    # SQLite 데이터베이스에 연결
    conn = sqlite3.connect(db_path)

    # 커서 객체 생성
    cur = conn.cursor()

    query = f"SELECT main, sub, title, id FROM men_clothes_info WHERE img_url_id IN ('{related_img_url}')"  
    # 쿼리 실행
    cur.execute(query)
    
    # 결과 가져오기
    clothes_name = cur.fetchall()

    query1 = 'SELECT * FROM men_clothes_info'

    cur.execute(query1)
    rows = cur.fetchall()

    # 컬럼 이름 가져오기
    col_names = [desc[0] for desc in cur.description]

    cur.close()
    conn.close()

    # 결과에서 문자열 값만 추출하여 출력
    for clothe_name in clothes_name:
        clothe_main = clothe_name[0]
        clothe_sub = clothe_name[1]
        clothe_title = clothe_name[2]
        clothe_id = clothe_name[3]

    with open(r'C:\\Users\\ITSC\\Desktop\\PMF_DB\\텍스트모델\\synonyms\\m_synonyms.json', 'r', encoding='utf-8') as f:
        m_synonyms = json.load(f)

    results = m_filter_mpnet_cos_items(rows, col_names, clothe_main, clothe_sub, clothe_title, synonyms = m_synonyms)
    
    context = {
        "related_img_url": related_img_url,
        "clothe_title" : clothe_title,
        "clothe_id" : clothe_id,
        "results" : results,
        }
    return render(request, 'clothes_detail.html', context)


def clothe_info_w(request, related_img_url):
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
    # SQLite 데이터베이스에 연결
    conn = sqlite3.connect(db_path)

    # 커서 객체 생성
    cur = conn.cursor()

    query = f"SELECT main, sub, title, id FROM women_clothes_info WHERE img_url_id IN ('{related_img_url}')"  
    # 쿼리 실행
    cur.execute(query)
    
    # 결과 가져오기
    clothes_name = cur.fetchall()

    query1 = 'SELECT * FROM women_clothes_info'

    cur.execute(query1)
    rows = cur.fetchall()

    # 컬럼 이름 가져오기
    col_names = [desc[0] for desc in cur.description]

    cur.close()
    conn.close()

    # 결과에서 문자열 값만 추출하여 출력
    for clothe_name in clothes_name:
        clothe_main = clothe_name[0]
        clothe_sub = clothe_name[1]
        clothe_title = clothe_name[2]
        clothe_id = clothe_name[3]
    
    with open(r'C:\\Users\\ITSC\\Desktop\\PMF_DB\\텍스트모델\\synonyms\\w_synonyms.json', 'r', encoding='utf-8') as f:
        w_synonyms = json.load(f)

    results = w_filter_mpnet_cos_items(rows, col_names, clothe_main, clothe_sub, clothe_title, synonyms = w_synonyms)

    context = {
        "related_img_url": related_img_url,
        "clothe_title" : clothe_title,
        "clothe_id" : clothe_id,
        "results" : results,
        }
    return render(request, 'clothes_detail.html', context)

def image_list(request, style=None):
    # SQLite 데이터베이스 파일 경로
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
        
    # SQLite 데이터베이스에 연결
    conn = sqlite3.connect(db_path)

    # 커서 객체 생성
    cur = conn.cursor()

    # 스타일에 따라 SQL 쿼리 구성
    if style:
        # 여러 스타일을 선택할 수 있도록 IN 연산자를 사용하여 쿼리를 작성합니다.
        styles = (style,)  # style을 튜플로 만듭니다.
        query = f"SELECT url FROM men_style_all WHERE style IN ({','.join(['?']*len(styles))}) AND style != '골프'"
        cur.execute(query, styles)
    else:
        query = "SELECT url FROM men_style_all"
        cur.execute(query)

    # 결과 가져오기
    results = cur.fetchall()

    style_url_list = []
    for result in results:
        style_url = result[0]
        style_url_list.append(style_url)

    cur.close()
    conn.close()

    paginator = Paginator(style_url_list, 60)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {'page_obj': page_obj, "style" : style}

    return render(request, './image_list.html', context)


def image_list_w(request, style=None):
    # SQLite 데이터베이스 파일 경로
    db_path = r'C:\Users\ITSC\Desktop\final_240607\test\PMF.db'
        
    # SQLite 데이터베이스에 연결
    conn = sqlite3.connect(db_path)

    # 커서 객체 생성
    cur = conn.cursor()

    # 스타일에 따라 SQL 쿼리 구성
    excluded_styles = ('골프', '아메카지', '미니멀', '고프고어', '레트로')  # 제외할 스타일 리스트
    if style:
        # 여러 스타일을 선택할 수 있도록 IN 연산자를 사용하여 쿼리를 작성합니다.
        styles = (style,)  # style을 튜플로 만듭니다.
        query = f"SELECT url FROM women_style_all WHERE style IN ({','.join(['?']*len(styles))}) AND style NOT IN ({','.join(['?']*len(excluded_styles))})"
        cur.execute(query, styles + excluded_styles)
    else:
        query = f"SELECT url FROM women_style_all WHERE style NOT IN ({','.join(['?']*len(excluded_styles))})"
        cur.execute(query, excluded_styles)

    # 결과 가져오기
    results = cur.fetchall()

    style_url_list = []
    for result in results:
        style_url = result[0]
        style_url_list.append(style_url)

    cur.close()
    conn.close()

    paginator = Paginator(style_url_list, 60)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {'page_obj': page_obj, "style" : style}

    return render(request, './image_list_w.html', context)

