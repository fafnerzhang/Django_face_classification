from django.shortcuts import render, redirect
from .form import UploadModelForm
from .models import Photo
from Face_recognition import Face_recognition
# Create your views here.

def index(request):

    photos = Photo.objects.all()
    form = UploadModelForm()
    face_recognize = Face_recognition(Photo,'unlabeled_image')

    if request.method == "POST":
        form = UploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('/photos')

    if request.GET.get('calculate_embedding'):
        face_recognize.check_new_file()
        face_recognize.face_embedding()

    if request.GET.get('Compare'):
        face_recognize.compare_image()
        return redirect('/result/')

    context = {
        'photos': photos,
        'form': form
    }

    return render(request, 'Image_upload/index.html', context)

def result(request):

    face_recognize = Face_recognition(Photo, 'unlabeled_image')
    target = Photo.objects.filter(tag=True)
    image_list = []
    for target_image in target:
        image_list.append([target_image,face_recognize.return_image(target_image.image)])

    context = {
        'target':target,
        'image_list':image_list,
    }

    if request.GET.get('return_to_upload'):
        return redirect('/photos/')

    return render(request, 'Image_upload/result.html', context)


'''def target(request):

    target = Target.objects.all()
    form = Upload_Target_Form()
    if request.method == 'POST':
        form = Upload_Target_Form(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            Check_new_file(Target, 'target_image')
            Face_embedding(Target, 'target_image')
            return redirect('/photos/')
    context = {
        'target': target,
        'form': form
    }

    return render(request, 'Image_upload/target.html',context)'''