from django.contrib import admin
from .models import Photo

class PhotoAdmin(admin.ModelAdmin):
    list_display = ('image','upload_date','embedding','target_tag','tag')

#class TargetAdmin(admin.ModelAdmin):
#    list_display = ('image','upload_date','name','embedding',)


admin.site.register(Photo,PhotoAdmin)
#admin.site.register(Target,TargetAdmin)
# Register your models here.
