from django.shortcuts import render, redirect
from .form import UploadModelForm, Upload_Target_Form
from .models import Photo, Target
from Face_recognition import Face_embedding, Check_new_file
# Create your views here.


def index(request):

    photos = Photo.objects.all()
    form = UploadModelForm()
    if request.method == "POST":
        form = UploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('/photos')

    context = {
        'photos': photos,
        'form': form
    }
    if request.GET.get('mybtn'):
        Face_embedding(Photo,'unlabeled_image')
    if request.GET.get('check_new_file'):
        print('check')
        Check_new_file(Photo)
    return render(request, 'Image_upload/index.html', context)

def target(request):

    target = Target.objects.all()
    form = Upload_Target_Form()
    if request.method == 'POST':
        form = Upload_Target_Form(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('/photos/')
    context = {
        'target': target,
        'form': form
    }

    return render(request, 'Image_upload/target.html',context)