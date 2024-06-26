# Generated by Django 5.0.6 on 2024-05-16 05:11

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="upload_image",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "upload_image",
                    models.FileField(blank=True, upload_to="Uploaded Files/%y/%m/%d/"),
                ),
                ("컬럼_업로드날짜", models.DateField(auto_now=True)),
            ],
        ),
    ]
