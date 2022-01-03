#!/usr/bin/env python
from __future__ import print_function
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model, model_from_json
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
import os
import cv2
from PIL import Image



import sys
from random import randrange as rr
if __package__ is None or __package__ == '':
    from os.path import basename
    from Compiler.classes.Utils import *
    from Compiler.classes.Compiler import *
else:
    from os.path import basename
    from Compiler.classes.Utils import *
    from Compiler.classes.Compiler1 import *


FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

dsl_path = "web-dsl-mapping.json"

def render_content_with_text(key, value):
    if FILL_WITH_RANDOM_TEXT:
        if key.find("card-header") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=rr(12, 16), space_number=rr(0, 2)))
        elif key.find("list-group-item") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=rr(18, 26), space_number=rr(0, 3)))
        elif key.find("large-title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=rr(11, 12), space_number=rr(0, 2)))
        elif key.find("link") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=rr(5, 13), space_number=0))
        elif key.find("button") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=rr(6, 9), space_number=0))
        elif key.find("footer") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  Utils.get_random_text(length_text=rr(32, 38), space_number=rr(1, 4), with_upper_case=False))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  Utils.get_random_text(length_text=rr(25, 35), space_number=rr(4, 8), with_upper_case=False))
    return value

def compileDSL(input_file):
    
    compiler = Compiler(dsl_path)

    file_uid = basename(input_file)[:basename(input_file).find(".")]
    path = input_file[:input_file.find(file_uid)]
    
    input_file_path = "{}{}.dsl".format(path, file_uid)
    output_file_path = "{}{}.html".format(path, file_uid)
    
    compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
word2idx ={'<END>': 9,
    '<START>': 7,
    'button': 2,
    'card-body': 21,
    'card-div': 18,
    'card-header': 20,
    'carousel': 19,
    'container': 8,
    'div-12': 22,
    'div-3': 10,
    'div-6': 15,
    'div-9': 23,
    'footer': 14,
    'img': 13,
    'jumbotron': 11,
    'large-title': 5,
    'link-list': 6,
    'list-group': 16,
    'list-group-item': 3,
    'navbar': 17,
    'row': 12,
    'text': 4,
    '{': 0,
    '}': 1}
idx2word = {0: '{',
 1: '}',
 2: 'button',
 3: 'list-group-item',
 4: 'text',
 5: 'large-title',
 6: 'link-list',
 7: '<START>',
 8: 'container',
 9: '<END>',
 10: 'div-3',
 11: 'jumbotron',
 12: 'row',
 13: 'img',
 14: 'footer',
 15: 'div-6',
 16: 'list-group',
 17: 'navbar',
 18: 'card-div',
 19: 'carousel',
 20: 'card-header',
 21: 'card-body',
 22: 'div-12',
 23: 'div-9'}


mobileNet = tf.keras.applications.MobileNetV2(weights="imagenet")
mobileNet.layers.pop()
mobileNet = Model(inputs=mobileNet.inputs, outputs=mobileNet.layers[-2].output)

def get_encoding(model, img):
    filename = img
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
        
    feature = model.predict(image, verbose=0)
    return feature
def predict_captions(image):
    start_word = ["<START>"]
    while True:
        par_caps = [(word2idx[i]) for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=94)
        #print(par_caps)
        preds = model.predict([np.array([image[0]]), np.array(par_caps)])
        #print(np.argmax(preds[0]))
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<END>" or len(start_word) > 94:
            break
            
    return ' '.join(start_word[1:-1])


app = Flask(__name__, template_folder='templates')
model = tf.keras.models.load_model('LR_MobileNet_LSTM_25epoch.h5')


@app.route('/')
def home():
	return render_template('index.html') 

@app.route('/sketch2code', methods = ['POST'])
def predict():
        print('predict')
        if request.method == 'POST':
            data = request.get_json()
            imagebase64 = data['image']
            print(imagebase64)
            imgbytes = base64.b64decode(imagebase64)
            with open("temp.jpg","wb") as temp:
                temp.write(imgbytes)
            image = Image.open('temp.jpg')
            
            new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
            new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
            new_image.convert('RGB').save('temp.jpg', "JPEG") 
            test_img = get_encoding(mobileNet, "temp.jpg")
            argmax = predict_captions(test_img)
            text_file = open("output.dsl", "w")
            n = text_file.write(argmax)
            text_file.close()
            compileDSL('output.dsl')
            print(argmax)
            return jsonify({
                'status': True 
            })
if __name__ == '__main__':
	app.run(debug=True)