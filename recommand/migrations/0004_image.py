# Generated by Django 5.0.6 on 2024-06-04 06:18

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("recommand", "0003_remove_upload_image_title"),
    ]

    operations = [
        migrations.CreateModel(
            name="Image",
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
                ("url", models.URLField()),
            ],
        ),
    ]
