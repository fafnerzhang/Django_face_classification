# Generated by Django 3.1.5 on 2021-01-29 11:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Image_upload', '0002_photo_embedding'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='tag',
            field=models.BooleanField(default=True),
        ),
    ]