

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import base64
import cv2


###################################################################################
###################################################################################
#								Sidebar 
st.sidebar.markdown(f"""
    # Examples!
    """)
##################################

st.sidebar.markdown(""" ### Example number 1""")
imagepath = 'raw_data/train_47.jpg'
st.sidebar.image(imagepath,use_column_width=False,output_format = 'JPEG')

st.sidebar.markdown('''
**Features:**
*primary rainforest, river, partly cloudy, road*
''')

##################################

st.sidebar.markdown(""" ### Example number 2 """)
imagepath = 'raw_data/train_258.jpg'
st.sidebar.image(imagepath,use_column_width=False,output_format = 'JPEG')

st.sidebar.markdown('''
**Features:**
*habitation, primary rainforest, road*
''')


###################################################################################
###################################################################################
#								INTRO

st.markdown("""# Green eye
## *Helping fight deforestation with deep learning*""")

st.markdown('### please upload a satellite image!')



###################################################################################    
###################################################################################
#								FUNCTIONS							

from keras import backend as K
def fbeta_score_K(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)  #saves the model so it wont have to load every f* time
def load():
	''' TO LOAD OUR MODEL'''
	model = load_model('Model-02-12',custom_objects={'fbeta_score_K': fbeta_score_K})  		#  '''dont target the model, target the FOLDER of the model'''
	return model
model = load()


def generate_imagedata(image):
	''' GENERATE THE DATA FROM THE LOADED IMAGE '''
	size = (128,128)    
	image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image = np.asarray(image)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img_resize = (cv2.resize(img, dsize=(128,128),    interpolation=cv2.INTER_CUBIC))/255.
	        
	return  img_resize[np.newaxis,...]	

def import_and_predict(image, model):
	'''MAKE THE PREDICTION '''
	return model.predict(image) 


#REMEMBER TO DELETE CLEAR WHEN YOU UPDATE YOUR MODE
#''' PRINT THE CODE'''
def decoder(prediction):
    l=[]
    alltags = [ 'clear', 'cloudy', 'haze', 'partly cloudy', 'agriculture', 'artisinal mine', 'bare ground', 'blooming', 
'blow down', 'cultivation', 'habitation', 'primary', 'road', 'selective logging', 'conventional mine', 
'slashu burn','water']
    for i in range(prediction.shape[1]):  #change this later and remove the clear function
        if prediction[0,i] > 0:
            l.append(alltags[i])
    classes = ", ".join(l)
    return classes
###################################################################################    
###################################################################################
#								lOAD THE IMAGE AND GET THE RESULTS	

file = st.file_uploader('',type = 'jpg',channels = 'RGB',)

if file is None:
    st.text("")

else:
	image = Image.open(file)
	st.image(image,use_column_width=False,output_format = 'JPEG')
	
	image = generate_imagedata(image)
	prediction = import_and_predict(image,model)
	#st.write(prediction)
	#st.write(type(prediction))
	
	st.write(f'### Features:\n  *{prediction}*')
	st.write(f'### Features:\n  *{decoder(prediction)}*')






