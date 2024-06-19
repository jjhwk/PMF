from django.contrib import admin
from recommand.models import Upload_image, Image


# Register your models here.
@admin.register(Upload_image)
class Upload_imageAdmin(admin.ModelAdmin):
    list_display = ["upload_image"]

@admin.register(Image)
class Image(admin.ModelAdmin):
    list_display = ["url"]
