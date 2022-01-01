# Jonathan Bryan - 2301852796
# Leonie Christina - 2301860406

import os
import cv2
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    train_names = os.listdir(root_path)

    return train_names

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

    train_image_list = []
    class_list = []

    for index, train_names in enumerate(train_names):
        full_name_path = root_path + '/' + train_names
        for image_path in os.listdir(full_name_path):
            full_image_path = full_name_path + '/' + image_path
            img = cv2.imread(full_image_path)

            train_image_list.append(img)
            class_list.append(index)

    return train_image_list, class_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    train_face_list = []
    train_class_list = []
    face_rect_list = []

    for index, image_list in enumerate(image_list):
        
        image_list = cv2.cvtColor(image_list, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(image_list, scaleFactor = 1.2, minNeighbors = 5)

        if len(detected_faces) < 1:
            continue
        if len(detected_faces) > 1:
            continue
        for face_rect in detected_faces:
            x,y,w,h = face_rect
            face_rects = x,y,w,h
            face_rect_list.append(face_rects)
            face_img = image_list[y:y+w, x:x+h]

            train_face_list.append(face_img)
            
            if(image_classes_list == None):
                continue
            train_class_list.append(image_classes_list[index])     

    return train_face_list, face_rect_list, train_class_list


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    test_image_list = []
    for image_path in os.listdir(test_root_path):
        full_image_path = test_root_path + '/' + image_path
        img_bgr = cv2.imread(full_image_path)

        test_image_list.append(img_bgr)
    return test_image_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    result_list = []

    for index, test_faces_gray in enumerate(test_faces_gray):
        result,_ = recognizer.predict(test_faces_gray)
        result_list.append(result)

    return result_list    

def get_verification_status(prediction_result, train_names, unverified_names):
    '''
        To generate a list of verification status from prediction results

        Parameters
        ----------
        prediction_result : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        unverified_names : list
            List containing all unverified names
        
        Returns
        -------
        list
            List containing all verification status from prediction results
    '''
    verification_status_list = np.empty(len(prediction_result)*2, dtype=object)
    test_length = len(prediction_result)

    for index, prediction_result in enumerate(prediction_result):
        for i in range(0, len(unverified_names)):
            
            if train_names[prediction_result] == unverified_names[i]:
                verification_status_list[index] = "Unverified"
                verification_status_list[index+test_length] = prediction_result
                break
            else:
                verification_status_list[index] = "Verified"
                verification_status_list[index+test_length] = prediction_result
    return verification_status_list

def draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results and verification status on the given test images

        Parameters
        ----------
        verification_statuses : list
            List containing all checked results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn
    '''
    test_length = len(test_image_list)
    drawn_image_list = np.empty(len(test_image_list)*2, dtype=object)
    vcount = 1
    uvcount = 0

    for index, test_image_list in enumerate(test_image_list):
        x,y,w,h = test_faces_rects[index]

        if verification_statuses[index] == "Unverified":
            cv2.putText(test_image_list, train_names[verification_statuses[index+test_length]], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

            cv2.rectangle(test_image_list, (x,y), (x+w, y+h), (0, 0, 255), 1)

            cv2.putText(test_image_list, verification_statuses[index], (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)  

            drawn_image_list[uvcount] = test_image_list
            uvcount+=2
        else:
            cv2.putText(test_image_list, train_names[verification_statuses[index+test_length]], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

            cv2.rectangle(test_image_list, (x,y), (x+w, y+h), (0, 255, 0), 1)

            cv2.putText(test_image_list, verification_statuses[index], (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            drawn_image_list[vcount] = test_image_list
            vcount+=2

    return drawn_image_list
    
def combine_and_show_result(image_list):
    '''
        To show the final image that already combined into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    length = len(image_list)*2

    for i in range(0, length):
        if i == 0:
            image_list[i] = cv2.resize(image_list[i], (250,250))
            uv_images = image_list[i]
        elif i == 1:
            image_list[i] = cv2.resize(image_list[i], (250,250))
            v_images = image_list[i]
        elif image_list[i] is not None :
            image_list[i] = cv2.resize(image_list[i], (250,250))
            if i%2==0:
                uv_images = cv2.hconcat([uv_images, image_list[i]]) 
            else:
                v_images = cv2.hconcat([v_images, image_list[i]])
        elif image_list[i+1] is None:
            break
        else:
            continue
    combined_image = cv2.vconcat([uv_images,v_images])
    cv2.imshow("Results", combined_image)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/Train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/Test"
    unverified_names = ['Raditya Dika', 'Anya Geraldine', 'Raffi Ahmad']

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    prediction_result = predict(recognizer, test_faces_gray)
    verification_statuses = get_verification_status(prediction_result, train_names, unverified_names)
    predicted_test_image_list = draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list) 