from django.contrib import admin
from .models import Photo, Target

class PhotoAdmin(admin.ModelAdmin):
    list_display = ('image','upload_date','embedding','tag')

class TargetAdmin(admin.ModelAdmin):
    list_display = ('image','upload_date','name','embedding',)


admin.site.register(Photo,PhotoAdmin)
admin.site.register(Target,TargetAdmin)
# Register your models here.
