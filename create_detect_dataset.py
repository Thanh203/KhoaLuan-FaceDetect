import cv2
import os
from ultralytics import YOLO
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import base64
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import shutil

save_directory = 'Converter'
id_base64_data = {}
harcascade = "haarcascade/haarcascade_frontalface_default.xml"

video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("http://10.159.223.174:8080/video")

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
print(width, height)
#Firebase storage
bucket_name = "attendance-86c04.appspot.com"
cred = credentials.Certificate('attendance-86c04-firebase-adminsdk-zzn2j-fefdb541b4.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': bucket_name
})

# Initialize Firebase Storage
bucket = storage.bucket()

def upload_file_to_storage(file_path, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)

def save_base64_data_to_file(id, base64_data):
    if id not in id_base64_data:
        id_base64_data[id] = [base64_data]
    else:
        id_base64_data[id].append(base64_data)
    file_path = os.path.join(save_directory, f"{id}.txt")
    with open(file_path, "a") as txt_file:
        txt_file.write(base64_data + ',\n')

def check_student_id_exists(id):
    files = bucket.list_blobs()
    for file in files:
        file_name = file.name
        if file_name.endswith('.txt'):
            nameId = str(file_name).split('/')[-1]
            nameId = nameId[:-4]
            if nameId == id:
                return False
    return True

def create_empty_flie(id):
    id_not_exists = check_student_id_exists(id)
    if id_not_exists:
        if id != '0' and id != '':
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            file_path = os.path.join(save_directory, f"{id}.txt")
            with open(file_path, "a") as txt_file:
                txt_file.write('')
            # Create empty file
            file_name_img = f"Students/{id}.txt"
            file_path_img = f"Converter/{id}.txt"
            blob = bucket.blob(file_name_img)
            blob.upload_from_filename(file_path_img)

segmentor = SelfiSegmentation()
bgimage = cv2.imread('Images/background.jpg')
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
bg_image_resized = cv2.resize(bgimage, (width, height))

def draw_bounding(x, y, w, h, index):
    x1 = x
    y1 = y 
    w1 = w + x 
    h1 = h + y  
    return x1, y1, w1, h1

def create_new_student_dataset(id):
    actions = ['straight', 'right', 'left', 'up']
    action_index = 0
    capture_count = 0
    count = 500
    action = 'Please look ' + actions[action_index]
    while capture_count < count :
        success, img = video_capture.read()
        if not success:
            break

        img = segmentor.removeBG(img, bg_image_resized, cutThreshold=0.8)
        
        cv2.putText(img,
                    f'{action}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_4)
        
        """if capture_count < count:
                    action = 'Please look ' + actions[action_index]
        if capture_count >= count:
                    action = 'Please look ' + actions[action_index]
        if capture_count >= count * 2:
                    action = 'Please look ' + actions[action_index]
        if capture_count >= count * 3:
                    action = 'Please look ' + actions[action_index]
        """

        facecascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Correct conversion code
        box = int(height/2 - 100)

        faces = facecascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=5, minSize=(box, box),flags=cv2.CASCADE_SCALE_IMAGE)

        # Find the largest face
        largest_face = None
        largest_area = 0

        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)

        # Draw the rectangle for the largest face only
        if largest_face is not None:
            (x, y, w, h) = largest_face
            x1, y1, x2, y2 = draw_bounding(x, y, w, h, action_index)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Crop the face from the frame
            cropped_face = img[y1:y2, x1:x2]
            #Converter from IMG to Base64 
            ret, buffer = cv2.imencode('.jpg', cropped_face)
            encoded_image = base64.b64encode(buffer).decode("utf-8")
            save_base64_data_to_file(id, encoded_image)
            print(capture_count)
            capture_count += 1

            # After capturing 20 images, move to the next action
            """if capture_count % count == 0 and not capture_count ==  count * 4:
                cv2.destroyAllWindows()
                while True:
                    continue_capture = input(f'The view on the {actions[action_index]} is completed. Please look {actions[action_index + 1]} to continue? (yes/no): ')
                    if continue_capture.lower() in ['yes', 'y']:
                        action_index += 1
                        break
                    elif continue_capture.lower() in ['no', 'n']:
                        return 
            """
            if capture_count ==  count:
                print(f'Completed writing data for {id}')
        
        cv2.imshow("Face", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    while True:
        id = input('mssv: ')
        id=id.upper()
        id_not_exists = check_student_id_exists(id)
        if id_not_exists:
            create_empty_flie(id)
            if id != '0' and id != '':
                create_new_student_dataset(id)
                file_name_img = f"Students/{id}.txt"
                file_path_img = f"Converter/{id}.txt"
                blob = bucket.blob(file_name_img)
                blob.upload_from_filename(file_path_img)
                if(True):
                    os.remove(f"Converter/{id}.txt")
            else:
                break
        else:
            print(f"The folder for student ID {id} exists.") 
        
    # Release the camera and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
    # Delete the Converter folder and its contents
    if os.path.exists('Converter'):
        shutil.rmtree('Converter')

if __name__ == "__main__":
    main()
    

