## ☞ [PMF - 딥러닝/머신러닝을 활용한 옷 추천 서비스]
- 개발언어 및 프레임워크 : `Python`, `SQL`, `Django`, `JavaScript`, `Html`, `CSS`
- 작업툴 : `Jupyter Notebook`, `Visual Studio Code`, `DBeaver`, `Notion`
- 사용 라이브러리 : `pandas`, `numpy`, `keras`, `torch`, `tensorflow`, `scipy`, `Resnet50`, `VGG19`,`konlpy`, `nltk`, `sentence_transformers`, etc.
- 인원 : 4명
- 기간 : 24.05.20 ~ 24.06.14
- 담당 역할 : 데이터 크롤링, 이미지 전처리, 이미지 스타일 다중분류, DB구현, 웹 구현
- 내용 :
  - 크롤링
    - 무신사 페이지에서 데이터를 수집하였습니다.
    - Selenium 과 BeautifulSoup 라이브러리를 사용하여 데이터를 동적 크롤링 하였습니다.
  - 데이터 전처리
    - 이미지 데이터
      - 모델이 이미지를 더 잘 학습 할 수 있도록 배경색을 검정색으로 통일하였습니다.
      - overfitting 을 방지하기 위해 카테고리별 이미지 데이터를 1000개로 맞추었습니다.
      - 과적합을 방지하고자 이미지 증강기법을 사용하였습니다. (이미지 좌우 반전, 이미지 x축 이동)
    - 텍스트 데이터
      - 모델이 데이터를 학습할 수 있도록 텍스트 데이터 임베딩을 하였습니다.(SBERT 모델사용)
      - 모델이 텍스트 분류시 동의어를 판단할 수 있도록 단어장을 만들었습니다.(20번 이상 나온 단어들만 사용) 
  - 모델링
    - 이미지 분류 모델
      - `이진 분류 모델`에서는 이미지가 옷인지 아닌지 판단할 수 있도록 학습시켰습니다. (CNN의 VGG19사용)
      - `다중 분류 모델`에서는 이미지가 어떤 스타일(ex. 캐주얼, 아메카지..)의 옷인지 판단할 수 있도록 학습시켰습니다. (CNN의 ResNet50 사용)
      - `유사도 분석 모델`에서는 입력된 이미지와 저장된 이미지들의 유사도를 비교하여 가장 높은 top10을 출력할 수 있도록 학습시켰습니다. (Cosine유사도, ResNet50 사용)
    - 텍스트 분류 모델
      - `유사도 분석 모델`에서는 입력된 텍스트와 저장된 임베딩 된 텍스트들의 유사도를 비교하여 가장 높은 top5를 출력할 수 있도록 학습시켰습니다. (Cosine유사도, SBert 사용)
  - DB 구성
    - 총 4개의 테이블로 구성 하였으며 남/여 각각 상세 `옷 id값`으로 상속되도록 구현하였습니다.
    - 남/여 이미지 데이터 정보 테이블
    - 남/여 상세 옷 정보 테이블
  - 웹 구성
    - [메인 페이지 ☞ 남/여 스타일 업로드 페이지 ☞ 스타일 정보 페이지 ☞ 옷 상세정보 페이지 ☞ 옷 구매 페이지] 순으로 넘어갈 수 있도록 구성하였습니다.
      
  - 전처리 기법 및 모델 사용 이유, 웹 페이지 구성의 보다 자세한 내용은 ☞ [요약 보고서](https://github.com/jjhwk/PMF/blob/main/PMF_.pdf) 에서 확인 하실 수 있습니다.
  
- 결과물 바로가기: ☞ [요약 보고서](https://github.com/jjhwk/PMF/blob/main/PMF_.pdf), ☞ [이미지모델 소스코드](https://github.com/jjhwk/PMF/blob/main/%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C/%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EB%AA%A8%EB%8D%B8%EB%A7%81/%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%8A%A4%ED%83%80%EC%9D%BC%EB%B3%84%20%EB%B6%84%EB%A5%98.ipynb),
          ☞ [이미지 유사도분석 소스코드](https://github.com/jjhwk/PMF/blob/main/%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C/%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EB%AA%A8%EB%8D%B8%EB%A7%81/%EC%9D%B4%EB%AF%B8%EC%A7%80_%EC%9E%84%EB%B2%A0%EB%94%A9_%EC%BD%94%EB%93%9C%EC%A0%95%EB%A6%AC_%ED%81%B4%EB%9E%98%EC%8A%A4%EB%A1%9C%EB%AC%B6%EA%B8%B0.ipynb), ☞ [웹 주요코드](https://github.com/jjhwk/PMF/blob/main/recommand/views.py)
