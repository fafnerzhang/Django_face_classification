from django import forms
from .models import Photo, Target

class UploadModelForm(forms.ModelForm):

    class Meta:
        model = Photo
        fields =('image',)
        widgets = {
            'image':forms.FileInput(attrs={'class':'form-control-file'})
        }


class Upload_Target_Form(forms.ModelForm):

    class Meta:
        model = Target
        fields = ('image','name')
        widgets = {
            'image':forms.FileInput(attrs={'class':'form-control-file','value':'選擇您的檔案'}),
            'name':forms.TextInput(attrs={'placeholder':'Input your name!'})
        }