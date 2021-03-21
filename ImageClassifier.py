#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:54:10 2021

@author: michaelantia

This is the end product of a Udemy course on creating an api using Flask to post
images for classification using a pre-trained MobileNet model from TensorFlow.
"""

from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

ALLOWED_EXTENSION = set(['txt','pdf','png','jpg','jpeg','gif'])
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3 

#os.chdir('../')

def allowed_file(filename):
    return '.' in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSION
    
app = Flask(__name__)
model = MobileNet(weights='imagenet', include_top=True)


#@app.route('/index')
@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    #Do sanity checks
    #Check whether post req has a file which has an img attribute
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should contain an attribute named image.')
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('ImageML.html', prediction='You did not select an image.')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) #rename file name in case there are weird chars
        print("***"+filename)
        
        #Preprocessing Steps
        x = []
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        # Need right size
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        # convert img to array
        x = image.img_to_array(img)
        # Expand the dims
        x = np.expand_dims(x, axis=0) # So that the array shape starts with (1,...)
        x = preprocess_input(x)
        pred = model.predict(x)
        lst = decode_predictions(pred, top=3)
        
        items = []
        for item in lst[0]:
            items.append({'name': item[1], 'prob': float(item[2])})
            
        response = {'pred': items}
        
        return render_template('ImageML.html', prediction = 'The image is most likely {}'.format(response))
    
    else:
        return render_template('ImageML.html', prediction = 'Invalid file extension.')
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
