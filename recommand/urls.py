from django.urls import path
from . import views

app_name = "recommand"

urlpatterns = [
    path("", views.main, name="main"),
    
    path("man/", views.man, name="man"),
    path("woman/", views.woman, name="woman"),

    path("img_detail/<path:img_url>/", views.img_detail, name="img_detail"),  # <path:img_url>로 변경
    path("img_detail_w/<path:img_url>/", views.img_detail_w, name="img_detail_w"),

    path("clothe_info/<path:related_img_url>/", views.clothe_info, name = "clothe_info"),
    path("clothe_info_w/<path:related_img_url>/", views.clothe_info_w, name = "clothe_info_w"),
    
    # 남자
    path("아메카지/", views.image_list, {'style': '아메카지'}, name="아메카지"),
    path("캐주얼/", views.image_list, {'style': '캐주얼'}, name="캐주얼"),
    path("시크/", views.image_list, {'style': '시크'}, name="시크"),
    path("댄디/", views.image_list, {'style': '댄디'}, name="댄디"),  
    path("비즈니스캐주얼/", views.image_list, {'style': '비즈니스캐주얼'}, name="비즈니스캐주얼"),
    path("고프고어/", views.image_list, {'style': '고프코어'}, name="고프코어"),  
    path("스포티/", views.image_list, {'style': '스포티'}, name="스포티"),  
    path("스트릿/", views.image_list, {'style': '스트릿'}, name="스트릿"),
    
    # 여기부터 여자
    path("캐주얼_w/", views.image_list_w, {'style': '캐주얼'}, name="캐주얼_w"),
    path("시크_w/", views.image_list_w, {'style': '시크'}, name="시크_w"),
    path("비즈니스캐주얼_w/", views.image_list_w, {'style': '비즈니스캐주얼'}, name="비즈니스캐주얼_w"),
    path("걸리시_w/", views.image_list_w, {'style': '걸리시'}, name="걸리시_w"),  
    path("로맨틱_w/", views.image_list_w, {'style': '로맨틱'}, name="로맨틱_w"),
    path("스포티_w/", views.image_list_w, {'style': '스포티'}, name="스포티_w"),  
    path("스트릿_w/", views.image_list_w, {'style': '스트릿'}, name="스트릿_w"),  
]
