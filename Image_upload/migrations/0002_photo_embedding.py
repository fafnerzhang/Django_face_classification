# Generated by Django 3.1.5 on 2021-01-29 07:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Image_upload', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='embedding',
            field=models.TextField(blank=True),
        ),
    ]
