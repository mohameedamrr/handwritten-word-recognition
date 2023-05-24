import imutils
import numpy as np
import cv2
from spellchecker import SpellChecker
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from keras import backend as K


def increaseImageQuality(img):
    # Check image type and convert if necessary
    if img.dtype != np.uint8:
        img = np.uint8(img / np.max(img) * 255)
    # Increase the contrast of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Sharpen the image
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img
def preprocessImage(img):
    (h, w) = img.shape

    # Calculate the new size while preserving the aspect ratio
    max_width = 256
    max_height = 64
    aspect_ratio = float(img.shape[1]) / img.shape[0]
    new_width = int(max_height * aspect_ratio)
    new_height = max_height

    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)

    # Ensure that the new height is at least 31 pixels
    if new_height < 31:
        new_height = 31
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    if h < max_height and w < max_width:
        resized_image = img
    else:
        resized_image = imutils.resize(img, width = 256, height = 64)

    # Calculate the padding values
    top = max(0, (h - new_height) // 2)
    bottom = max(0, h - new_height - top)
    left = max(0, (w - new_width) // 2)
    right = max(0, w - new_width - left)

    # Add padding if necessary
    resized_image_with_padding = resized_image
    #resized_image_with_padding = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Create a blank white image with the final size
    final_img = np.ones([64, 256])*255 

    # Resize the padded image to fit the final size
    resized_padded_image = cv2.resize(resized_image_with_padding, (256, 64), interpolation=cv2.INTER_AREA)

    # Copy the resized padded image to the final image
    final_img[:resized_padded_image.shape[0], :resized_padded_image.shape[1]] = resized_padded_image
    final_img = increaseImageQuality(final_img)
    # Rotate the final image 90 degrees clockwise
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('images//Model Temp Images/test.png',final_img)
    return final_img

def cropImage(im):
    # Get rid of existing black border by flood-filling with white from top-left corner
    ImageDraw.floodfill(im,xy=(0,0),value=(255,255,255),thresh=10)

    # Get bounding box of text and trim to it
    bbox = ImageOps.invert(im).getbbox()
    trimmed = im.crop(bbox)

    # Add new white border, then new black, then new white border
    res = ImageOps.expand(trimmed, border=1, fill=(255,255,255))
#     res = ImageOps.expand(res, border=5, fill=(255,255,255))
#     res = ImageOps.expand(res, border=5, fill=(255,255,255))
    res.save('images/Model Temp Images/result2.png')
    return res




def label_to_num(label):
    alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
    max_str_len = 34 # max length of input labels
    num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
    num_of_timestamps = 64 # max length of predicted labels
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
    max_str_len = 34 # max length of input labels
    num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
    num_of_timestamps = 64 # max length of predicted labels
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def autocorrect(prediction):
    my_file = open("WordsDictionary.txt", "r")
    # reading the file
    data = my_file.read()

    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.split("\n")
    #print(data_into_list)
    spell = SpellChecker()
    spell.word_frequency.load_words(data_into_list)
    word = spell.correction(prediction)
    word = word.upper()
    return word

def processOutsideImage(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

    # Create horizontal kernel then dilate to connect text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours and filter out noise using contour approximation and area filtering
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h)
        if area > 1200 and area < 50000 and ar < 8:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            # Crop the image based on the bounding box
            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite("images/Model Temp Images/result2.png", cropped_image)

    # Bitwise-and input image and mask to get result
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(image, image)
    # result[mask==0] = (255,255,255) # Color background white

    # NEW CODE HERE TO END _____________________________________________________________
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2
    kernel = np.ones((5, 191), np.uint8)
    close = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

    # get external contours
    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]


    # draw contours
    result2 = result.copy()
    for cntr in contours:
        # get bounding boxes
        pad = 100
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result2, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 4)
        

    i = 0
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # to save the images
        try:
            cv2.imwrite('images/Model Temp Images/result2.png', image[y-50:y+h+50,x-50:x+w+50])
        except:
            continue
        i += 1


def brightenOutsideImage(img):
    # increase contrast
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 400

    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations = 3)
    cv2.imwrite("images/Model Temp Images/test2.png", imgMorph)
    return imgMorph



def getPredictedWord():
    model = tf.keras.models.load_model('Model/new_model.h5')

    image = Image.open('images/Model Temp Images/captured_snapshot.jpg')
    
    image = cropImage(image)
    
    image = cv2.imread('images/Model Temp Images/captured_snapshot.jpg', cv2.IMREAD_GRAYSCALE)
    image = preprocessImage(image)
    
    
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))

    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])

    predicted_word = autocorrect(num_to_label(decoded[0]))

    return predicted_word

def getPredictedOutsideWord():
    model = tf.keras.models.load_model('Model/new_model.h5')
    image = cv2.imread("images/Model Temp Images/captured_snapshot_out.jpg")
    image = processOutsideImage(image)
    img = cv2.imread('images/Model Temp Images/result2.png', cv2.IMREAD_GRAYSCALE)
    brightenOutsideImage(img)
    image = cv2.imread('images/Model Temp Images/test2.png', cv2.IMREAD_GRAYSCALE)
    image = preprocessImage(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))

    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])

    predicted_word = autocorrect(num_to_label(decoded[0]))
    
    return predicted_word