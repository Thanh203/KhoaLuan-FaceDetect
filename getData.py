from firebase_admin import credentials, initialize_app, storage
import base64
from io import BytesIO
from PIL import Image
import os
import shutil

import zipfile

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Để bảo lưu cấu trúc thư mục, ta cần thêm thư mục gốc
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

def download_files(bucket_name):
    
    cred = credentials.Certificate(r"attendance-86c04-firebase-adminsdk-zzn2j-fefdb541b4.json")
    initialize_app(cred, {'storageBucket': bucket_name})

    bucket = storage.bucket()

    files = bucket.list_blobs()

    arr_file_names = []

    label=0
    if not files:
        print("The file is empty or the connection is lost, please check again")
    else:
        for file in files:
            # Lấy tên file
            file_name = file.name

            if file_name.endswith('.txt'):
                # Đọc nội dung của file
                file_content = file.download_as_string()
                
                # Thêm tên file vào mảng
                original_filename_arr = str(file_name).split('/')[-1]
                arr_name=str(original_filename_arr).split('.')
                arr_file_names.append(str(arr_name[0]))

                file_content=file_content.decode('utf-8')

                lst_img=str(file_content).split(',')

                save_ImgTrain=f'{path}/train/images'
                os.makedirs(save_ImgTrain, exist_ok=True)

                save_ImgVal=f'{path}/val/images'
                os.makedirs(save_ImgVal, exist_ok=True)

                save_LabelTrain=f'{path}/train/labels'
                os.makedirs(save_LabelTrain, exist_ok=True)

                save_LabelVal=f'{path}/val/labels'
                os.makedirs(save_LabelVal, exist_ok=True)

                for i in range(len(lst_img)):
                    if lst_img[i].strip():

                        img_bytes = base64.b64decode(lst_img[i])

                        # Mở ảnh từ bytes
                        img = Image.open(BytesIO(img_bytes))
                        img=img.convert('RGB')

                        # Lấy tên file gốc từ đường dẫn trên Firebase
                        original_filename = str(file_name).split('/')[-1]
                        img_name=str(original_filename).split('.')

                        label_filename = f'{img_name[0]}-{i}.txt'

                        if i % 3 != 0:
                            # Lưu ảnh với tên file gốc và số thứ tự, không bao gồm đường dẫn trên Firebase
                            img.save(os.path.join(save_ImgTrain, f'{img_name[0]}-{i}.png'))
                            label_path = os.path.join(save_LabelTrain, label_filename)
                        else:
                            img.save(os.path.join(save_ImgVal, f'{img_name[0]}-{i}.png'))
                            label_path = os.path.join(save_LabelVal, label_filename)
                        
                        with open(label_path, 'w') as label_file:
                            label_file.write(f"{label} 0.5 0.5 1 1")

                label+=1

    
        train_file = f'{path}/data.yaml'
        f = open(train_file, 'w')
        f.write(f"train: ../{path}/train/images\n")
        f.write(f"val: ../{path}/val/images\n")
        f.write(f"nc: {len(arr_file_names)}\n")
        f.write("names: [")
        for id in arr_file_names:
            f.write(f"{id} ")
            if (arr_file_names[len(arr_file_names) - 1] != id):
                f.write(', ')
        f.write(']')


path='dataset_detect2'

# Delete the Converter folder and its contents
if os.path.exists(f'{path}'):
    shutil.rmtree(f'{path}')

bucket_name = "attendance-86c04.appspot.com"

download_files(bucket_name)

folder_path = f'{path}'
output_path = f'{path}.zip'
zip_folder(folder_path, output_path)
