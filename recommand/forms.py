from django.forms import ModelForm
from recommand.models import Upload_image
from django import forms

class FileUploadForm(forms.Form):
    image = forms.ImageField()