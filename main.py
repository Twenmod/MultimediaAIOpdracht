from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import keyboard
#settings

confidence_treshold = 90 # amount of confidence before counting the gesture as true
hold_duration = 50 # amount of time to have to hold a gesture before output
input_timeout = 100 # amount of time in ms to timeout after giving output




#vars

tresholds = [0,0,0,0,0,0,0,0]

#multimedia output

def outputMedia(selection = 0):

    if selection == 0: #Background case
        return
    elif selection == 1: #ThumbRight
        keyboard.press_and_release("VK_MEDIA_PLAY_PAUSE")
        return


#AI


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

class_name = ""
index = 0
confidence_score = 0

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    img = cv2.putText(image, class_name, (0,185), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 2, cv2.LINE_AA)
    img = cv2.putText(image, str(tresholds[index]), (0,200), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow("Webcam Image", img)


    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[0:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    
    #Gesture calculation
    
    #decrease all tresholds
    i = 0
    for treshold in tresholds:
        treshold -= 1

        if (treshold > hold_duration):
            #output gesture and reset all and add a small delay to avoid reinput
            outputMedia(i)
            tresholds = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            time.sleep(input_timeout/1000)


        i+=1

    if (confidence_score*100)>confidence_treshold:
        #enough confidence in the gesture
        tresholds[index] += 2


 


    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

    time.sleep(0.01)

camera.release()
cv2.destroyAllWindows()





