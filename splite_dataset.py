# import os
# import torch
# from torchvision import datasets
# from torchvision.utils import save_image
# from torchvision.transforms import ToTensor
# import tqdm
# from PIL import Image
# import math

# def remove_other_file(path):
#     type = ['.jpg','.jpeg']
#     for sub_dir,_,files in os.walk(path):
#         for file in files:
#             if os.path.splitext(file)[1].lower() not in type:
#                 os.remove(os.path.join(path,file))

# def jpg_format(path):
#     file_type = ['.jpeg','.jpg']
#     for sub_dir,_,files in os.walk(path):#获取目录下所有子目录和文件
#         print("the sub_dir is :",sub_dir)
#         for file in files:
#             if os.path.splitext(file)[1].lower() not in file_type:#获取扩展名转为小写
#                 with Image.open(os.path.join(path,file))as image:
#                     image = image.convert('RGB')
#                     file = image.save(os.path.join(path,os.path.splitext(file)[0]+'.jpg'),"JPEG")
#     return "转换成功"

# def splite(path,train_ratio,test_ratio,image_format,name):
#     data = datasets.ImageFolder(path,transform=ToTensor())
#     #class_to_idx = {'dogs': 0, 'cats': 1, 'rabbits': 2}
#     image = list(data.class_to_idx.keys())
#     image = ['MildDemented', 'ModerateDemented','NonDemented']
#     #print("the image is :",image)
#     datalens = len(data)
#     trainlens = math.ceil(datalens*train_ratio)
#     vallens = min(datalens-trainlens,math.ceil(datalens*test_ratio))
#     print("the three len is :",datalens,trainlens,vallens)
#     loader = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)
#     for file in image:
#         if not os.path.isdir(os.path.join(path,"train",file)):
#             os.makedirs(os.path.join(path,"train",file))
#         if not os.path.isdir(os.path.join(path,"val",file)):
#             os.makedirs(os.path.join(path,"val",file))
    
#     for index,file in tqdm.tqdm(enumerate(loader)):
#         images,label = file
#         #print("the len is :",index)
#         #file 是 loader 返回的数据批次。file 通常是一个包含两部分的元组：第一部分是 images（图片批次）。第二部分是 labels（对应图片的标签）
#         while trainlens > 0:
#             #save_image(images,os.path.join(path,'train',image[label],str(index+1)+'.'+image_format))
#             save_image(images,os.path.join(path,'train',image[label],f"{name}{str(index)}.{image_format}"))
#             trainlens-=1
#             break
#         while vallens >0 :
#             #save_image(images,os.path.join(path,'val',image[label],str(index+1),'.'+image_format))
#             save_image(images,os.path.join(path,'val',image[label],f"{name}{str(index)}.{image_format}"))
#             vallens-=1
#             break
# if __name__ =="__main__":
#     splite(path="/home/bygpu/med/med2d/AugmentedAlzheimerDataset",train_ratio=0.8,test_ratio=0.2,image_format="jpg",name="ct_healthy")

import os
import torch
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
import tqdm
from PIL import Image
import math

def remove_other_file(path):
    type = ['.jpg', '.jpeg']
    for sub_dir, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() not in type:
                os.remove(os.path.join(sub_dir, file))

def jpg_format(path):
    file_type = ['.jpeg', '.jpg']
    for sub_dir, _, files in os.walk(path):
        print("the sub_dir is :", sub_dir)
        for file in files:
            if os.path.splitext(file)[1].lower() not in file_type:
                with Image.open(os.path.join(sub_dir, file)) as image:
                    image = image.convert('RGB')
                    image.save(os.path.join(sub_dir, os.path.splitext(file)[0] + '.jpg'), "JPEG")
    return "转换成功"

def splite(path, train_ratio, test_ratio, image_format, name):
    data = datasets.ImageFolder(path, transform=ToTensor())
    image = list(data.class_to_idx.keys())
    
    image = ['Healthy', 'Tumor','TumorBig']
    datalens = len(data)
    trainlens = math.ceil(datalens * train_ratio)
    vallens = min(datalens - trainlens, math.ceil(datalens * test_ratio))
    print("the three len is :", datalens, trainlens, vallens)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    for file in image:
        if not os.path.isdir(os.path.join(path, "train", file)):
            os.makedirs(os.path.join(path, "train", file))
        if not os.path.isdir(os.path.join(path, "val", file)):
            os.makedirs(os.path.join(path, "val", file))

    for index, (images, label) in tqdm.tqdm(enumerate(loader)):
        # 将 label 从张量转换为整数
        label = label.item()
        # 限制 label 的范围在 0 - 2 之间
        label = max(0, min(label, 2))

        if index < trainlens:
            try:
                save_image(images, os.path.join(path, 'train', image[label], f"{name}{str(index)}.{image_format}"))
            except Exception as e:
                print(f"Error saving image to train set: {e}")
        elif index < trainlens + vallens:
            try:
                save_image(images, os.path.join(path, 'val', image[label], f"{name}{str(index)}.{image_format}"))
            except Exception as e:
                print(f"Error saving image to validation set: {e}")

if __name__ == "__main__":
    splite(path="/home/bygpu/med/med2d/Brain2D/Brain Tumor CT scan Images", train_ratio=0.8, test_ratio=0.2, image_format="jpg", name="ct_healthy")