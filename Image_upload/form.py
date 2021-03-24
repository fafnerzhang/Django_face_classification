from django import forms
from .models import Photo


class UploadModelForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields =('image','tag')
        widgets = {
            'image':forms.FileInput(attrs={'class':'form-control-file'}),
            'tag':forms.NullBooleanSelect(attrs={'class':'from-control','value':'True'})
        }


'''class Upload_Target_Form(forms.ModelForm):

    class Meta:
        model = Target
        fields = ('image','name')
        widgets = {
            'image':forms.FileInput(attrs={'class':'form-control-file','value':'選擇您的檔案'}),
            'name':forms.TextInput(attrs={'placeholder':'Input your name!'})
        }'''