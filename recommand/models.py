from django.db import models

# Create your models here.

class Upload_image(models.Model) :
    upload_image = models.ImageField("이미지", upload_to = "media/images/", blank = True)


class Image(models.Model):
    url = models.URLField(max_length=200)
    # Add other fields as needed

    def __str__(self):
        return self.url
