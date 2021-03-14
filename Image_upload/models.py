from django.db import models
from django.utils import timezone


class Photo(models.Model):
    image = models.ImageField(upload_to='image/', blank=False, null=False)
    upload_date = models.DateField(default=timezone.now)
    embedding = models.TextField(blank=True)
    tag = models.BooleanField(default=False)


class Target(models.Model):
    image = models.ImageField(upload_to='target_image/', blank=False, null=False)
    upload_date = models.DateField(default=timezone.now)
    embedding = models.TextField(blank=True)
    name = models.CharField(max_length=50)

# Create your models here.
