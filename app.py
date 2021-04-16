
import os
import sys
import io
import time,numpy
# from dotenv import load_dotenv
import flask
from flask import Flask, render_template, request, redirect, url_for#, session
import pickle
import PIL.Image
import base64
from threading import Lock
import json
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator
import pandas
import tensorflow
# import tensorflow.keras.layers
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# from sklearn.model_selection import train_test_split

from tensorflow.keras.models import model_from_json

json_file = open('weights/modelage.json', 'r')
model_json = json_file.read()
json_file.close()
model_age = model_from_json(model_json)
model_age.load_weights("weights/modelage.h5")

json_file = open('weights/modelethnicity.json', 'r')
model_json = json_file.read()
json_file.close()
model_ethnicity = model_from_json(model_json)
model_ethnicity.load_weights("weights/modelethnicity.h5")

json_file = open('weights/modelgender.json', 'r')
model_json = json_file.read()
json_file.close()
model_gender = model_from_json(model_json)
model_gender.load_weights("weights/modelgender.h5")

# g_Gs = None
# g_Synthesis = None
# g_Lpips = None
# g_Projector = None
# g_Session = None
# g_LoadingMutex = Lock()
# pngfiles= {}
# alreadysent=set()
gen = StyleGan2Generator(weights='ffhq' , impl='ref', gpu=False)
w_avg = np.load('weights/{}_dlatent_avg.npy'.format('ffhq' ))
	
def loadGs():
	with g_LoadingMutex:
		global g_Gs, g_Synthesis
		if g_Gs: return g_Gs, g_Synthesis
		global model_name
		global g_Session
		if g_Session is None:
			dnnlib.tflib.init_tf()
			g_Session = tf.get_default_session()
		with open(model_path, 'rb') as f:
			with g_Session.as_default():
				Gi, Di, Gs = pickle.load(f)
				g_Gs = Gs
				global g_dLatentsIn
				g_dLatentsIn = tf.placeholder(tf.float32, [Gs.components.synthesis.input_shape[1] * Gs.input_shape[1]])
				dlatents_expr = tf.reshape(g_dLatentsIn, [1, Gs.components.synthesis.input_shape[1], Gs.input_shape[1]])
				g_Synthesis = Gs.components.synthesis.get_output_for(dlatents_expr, randomize_noise = False)
	return g_Gs, g_Synthesis

# def regenerate(gen, seed, w_avg, truncation_psi=1):
# 	dlatents = w_avg + (gen.mapping_network(np.random.RandomState(seed).randn(1, 512).astype('float32')) - w_avg) * 0.5
# 	return convert_images_to_uint8(gen.synthesis_network(dlatents), nchw_to_nhwc=True, uint8_cast=True)

app = flask.Flask(__name__, static_url_path = '', static_folder = './static')

# @app.route('/startgenerator', methods=['GET'])
# def generator():
# 	global g_Session
# 	global g_dLatentsIn
# 	global fav
# 	global pngfiles
# 	global alreadysent
# 	gs, synthesis = loadGs()
# 	size=200
# 	fmt = dict(func = dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc = True)
# 	with g_Session.as_default():
# 		regenerate(generator, seed=96, w_avg=w_average, truncation_psi=0.5)

# 		# while True:

# 		# 		encode=latentCode.encodeFloat32(mutatedarr).decode()
# 		# 		images = gs.run(np.array([mutatedarr]), None, truncation_psi = latentt[0], randomize_noise = 0 != 0, output_transform = fmt)
# 		# 		pngfiles[encode]=io.BytesIO()

# 		# 		buffer=PIL.Image.fromarray(images[0], 'RGB').resize((size,size),PIL.Image.NEAREST)
# 		# 		buffer.save(pngfiles[encode], PIL.Image.registered_extensions()['.png'])
# 		# 		# PIL.Image.fromarray(images[0], 'RGB').resize((size,size),PIL.Image.NEAREST).save(pngfiles[encode], PIL.Image.registered_extensions()['.png'])
# 		# 		# with open("o.txt","wb") as f:
# 		# 		# 	f.write()
# 		# 		print('generation', t3- t2, "iobuffer",t5- t4,"stack:",len(set(pngfiles.keys())),len(set(pngfiles.keys())-alreadysent))


@app.route('/home', methods=['GET'])
def home():
	for filename in os.listdir('static'):
		file_path = os.path.join('static', filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			pass

	return flask.Response(render_template("image.html"))


@app.route('/', methods=['GET'])
def jpg():
	global gen
	global w_avg
	dlatents = (gen.mapping_network(np.random.RandomState(np.random.randint(0,30000)).randn(1, 512).astype('float32'))) * ((numpy.random.rand()*0.6)+0.3)
	image = convert_images_to_uint8(gen.synthesis_network(dlatents+np.multiply(np.random.rand(512),0.3)), nchw_to_nhwc=True, uint8_cast=True)

	# x = io.BytesIO()#resize((200,200),PIL.Image.NEAREST).
	img = PIL.Image.fromarray(numpy.array(image[0]), 'RGB')
	# img.save(x, PIL.Image.registered_extensions()['.jpg'])
	# pic.putdata(list(tuple(pixel) for pixel in pix))
	# xx=np.array(img.resize((48,48),PIL.Image.NEAREST).convert('L'))
	# xx=np.array(xx/255, dtype="float32").reshape(1,48,48,1)#/255
	# xx=(xx/255).reshape(1,48,48,1)
	# image=regenerate(generator, seed=96, w_avg=w_average, truncation_psi=0.5)
	# return flask.Response(x.getvalue(), mimetype='image/jpg')
	# print(xx)
	xx= np.array([([jj[1] for jj in ii[21::21]]) for ii in image[0]][21::21])
	xx=np.array(xx, dtype="float32").reshape(1,48,48,1)#/255
	m=list(model_ethnicity.predict(xx))
	ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][m.index(max(m))]
	age = int(model_age.predict(xx)[0][0])//100
	gender = 'Male' if round(model_gender.predict(xx)[0][0]) else "Female"
	details=(f'Prediction: {age} year old {ethnicity} {gender}')

	image_name = "image/" + str(time.time()) + ".jpg"

	# for filename in os.listdir('static/'):
	# 	 if filename.startswith('image_'):  # not to remove other images
	# 		 os.remove('static/' + filename)
	with open('static/' + image_name, 'w') as f:
		img.save(f, PIL.Image.registered_extensions()['.JPEG'])

	return flask.render_template_string(
		"""<html>
			  <head>
			  	<title>AAA</title>
			  </head>
			   <body>
				 <h1>{{details}}</h1>
				  <img src="{{ url_for('static', filename=image_name) }}" style='height:88vh' />
			   </body>
			</html>
			""",details=details, image_name=image_name)

if __name__ == "__main__":
	# argv=sys.argv
	# ipaddress=socket.gethostbyname(socket.gethostname())#"0.0.0.0",os.getgev('HTTP_ADDR')
	try: app.run(port = 5000, host = "0.0.0.0", threaded = True,debug=False)#os.getenv('HTTP_HOST')  ,log = Noneos.getenv('HTTP_PORT')
	except:	print('server interrupted:', sys.exc_info())
