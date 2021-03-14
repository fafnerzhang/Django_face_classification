from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import shutil
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os, shutil
from Image_upload.models import Photo

def Face_embedding(model,folder_name):
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

    aligned = []
    names = []
    name = []
    face_num_list = []
    for x, y in loader:
        face_num = 0
        x_aligned, prob = mtcnn(x, return_prob=True)

        if x_aligned is not None:
            for number in prob:
                print(f'Face detected with probability: {number}')
            for num in range(x_aligned.shape[0]):
                aligned.append(x_aligned[num])
                name.append(dataset.idx_to_class[y])
                face_num += 1
        else:
            aligned.append(torch.zeros([3,160,160]))
            print('no detect')
            name.append(dataset.idx_to_class[y])
            face_num += 1
        face_num_list.append(face_num)
    print('face num list:',face_num_list)
    names = names + Load_image_name(folder_name,face_num_list) # all image name
    print('names:',names)
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()
    print('embedding shape',embeddings.shape)
    print('names shape',len(names))
    #embeddings_dict = { names[i]:embeddings[i] for i in range(len(names))}
    embedding_dict = {}
    now = 0
    for j in face_num_list:
        embedding_dict[f'{names[now]}'] = embeddings[now:now+j,:]
        print('embedding_dict shape:',embedding_dict[f'{names[now]}'].shape)
        now += j
    print('embedding dict',embedding_dict)
    embedding_save(model,embedding_dict)
    image_path = os.path.join(BASE_DIR,'image')
    shutil.rmtree(image_path)
    os.mkdir(image_path)

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=names, index=names))
    for image in Photo.objects.all():
        print(' model _ ',image.image,image.embedding,type(image.embedding))
        image_tensor = image.embedding
        print(image_tensor,type(image_tensor))
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

def Load_target_name(target):
    path = os.path.join(os.getcwd(),'media',target)
    target_name = os.listdir(path)
    return target_name

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

def Check_new_file(image):
    photos = image.objects.all()
    path = os.path.join(os.getcwd(),'media')

    New_image_list = []
    for image in photos:
        #print(f'image name:{image.image} "{image.embedding}"')
        if image.embedding is "" :
            image_name = os.path.join(path,str(image.image))
            New_image_list.append(image_name)
            shutil.copyfile(image_name,image_name.replace('image','unlabeled_image/image'))
            print(image_name,'is new')

    return New_image_list