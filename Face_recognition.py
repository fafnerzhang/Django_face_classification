from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import shutil
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os, shutil, json
from Image_upload.models import Photo

def Face_embedding(model,folder_name):
    '''
    Recognize face in the model's image and save the embedding to the model.

    :param model: The target model
    :param folder_name: Folder Path of the model image
    :return:
    '''
    BASE_DIR = os.path.join(os.getcwd(),'media',folder_name)
    workers = 0 if os.name =='nt' else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device,keep_all=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(BASE_DIR)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []    # aligned tensor of image, element shape:[3,160,160]
    names = []  # list of image names
    face_num_list = []  # face number of each image

    for x, y in loader:
        face_num = 0
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:   # if mtcnn detected face ,adding face tensor of to aligned
            for number in prob:
                print(f'Face detected with probability: {number}')
            for num in range(x_aligned.shape[0]):
                aligned.append(x_aligned[num])
                face_num += 1
        else:                       # if no face in the image adding 0 tensor to aligned
            aligned.append(torch.zeros([3,160,160]))
            print('no detect')
            face_num += 1
        face_num_list.append(face_num)

    names = names + Load_image_name(folder_name,face_num_list) # Get all image name in the folder

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()     # calculate face embedding of each face

    embedding_dict = {}
    now = 0
    for j in face_num_list:        # create a dict of {image_name:face_tensor(shape of [face_num,512])}
        embedding_dict[f'{names[now]}'] = embeddings[now:now+j,:]
        now += j

    embedding_save(model,embedding_dict)        # save the embedding tensor to the model

    image_path = os.path.join(BASE_DIR,'image')
    shutil.rmtree(image_path)   # delete classify image
    os.mkdir(image_path)

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]       # distance table of this image group
    print(pd.DataFrame(dists, columns=names, index=names))


def Compare_image(model):
    '''

    :param target: target image model
    :param model: whole image model
    :return:
    '''
    target = model.objects.filter(tag=True)
    photo = model.objects.all()
    for target_image in target:
        print(f'\n{target_image.image} start comparing \n')
        for image in photo:
            target_tensor = torch.tensor(json.loads(target_image.embedding))
            image_tensor = torch.tensor(json.loads(image.embedding))
            for i in range(image_tensor.size(0)):
                threshold = (image_tensor[i]-target_tensor).norm().item()
                print(f' {image.image} round "{i}" threshold:',threshold)

                if threshold <= 1 :
                    pic = model.objects.get(image=f'{str(image.image)}')
                    pic.target_tag = pic.target_tag + f' {target_image.image}'
                    pic.save()
                    print(f'{image.image} save. tag: {image.target_tag}')
                else :
                    print(image.image,' are not target')
    return True


def Return_image(target_name):
    target = Photo.objects.filter(target_tag__contains=target_name)
    return target


def Load_image_name(folder_name,face_num):
    '''
    Get whole image name in file
    :param model_name:
    :return:
    '''
    path = os.path.join(os.getcwd(),'media',folder_name,'image')
    photo_list = os.listdir(path)
    for i in range(len(photo_list)) :
        photo_list[i] = 'image/'+ photo_list[i]
    print(photo_list)
    image_list = []
    count = 0
    for num in face_num:
        for i in range(num):
            image_list.append(photo_list[count])
        count += 1
    return image_list



def embedding_save(image,embedding_dict):
    '''

    :param image: model object
    :param embedding_dict: with final tensor of new picture

    Save new object to django model

    :return:
    '''

    photos = image.objects.all()
    for image in photos:
        try:
            pic = photos.filter(image=f'{str(image.image)}')
            pic.update(embedding=str(embedding_dict[f'{str(image.image)}'].tolist()))

        except:
            continue

    return True

def Check_new_file(image,folder_name):

    photos = image.objects.all()
    path = os.path.join(os.getcwd(),'media')

    for image in photos:
        #print(f'image name:{image.image} "{image.embedding}"')
        if image.embedding is "" :
            image_name = os.path.join(path,str(image.image))
            shutil.copyfile(image_name,image_name.replace('image',folder_name+'/image'))
            print(image_name,'is new')

    return True