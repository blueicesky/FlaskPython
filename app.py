from flask import Flask, redirect, render_template, request, session, url_for, send_from_directory, jsonify
from nn_model import NN_model
from rr_model import RR_model
import copy as copy
import traceback
from datetime import datetime
from werkzeug import secure_filename
import pickle as pkl
import os
from configparser import SafeConfigParser
from logger import Logger

#Logger for logging capabilities
log = Logger('main_app.log')
log.info('Starting Application')
log.info('Reading variables from config file.')
config = SafeConfigParser()
config.read('config.ini')
UPLOAD_FOLDER = config.get('UploadFolders','UPLOAD_FOLDER_PRED')
UPLOAD_FOLDER_TRAIN = config.get('UploadFolders','UPLOAD_FOLDER_TRAIN')
DOWNLOAD_FOLDER = config.get('DownloadFolders','DOWNLOAD_FOLDER')
ALLOWED_EXT = set(['csv'])

log.info('Setting Flask app initial variables')
#Initializing flask variable here
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Secret key is MANDATORY!
app.secret_key = os.urandom(12)
#Declaring model
log.info('Setting up model variables.')
model = None
temp_model = None
secondary_model = None
fname = 'nn_model.pkl'

#Checking to see if the pickle file exists
if os.path.isfile(fname):
	#Load the pickle file if it exists
	log.info('Pickle file detected, initializing loading sequence.')
	file_opened = open(fname, "rb")
	model = pkl.load(file_opened)
	file_opened.close()
	log.info('Pickle file loaded successfully')
else:
	#No pickle file found
	log.info('No pickle file found. Creating new model class.')
	model = NN_model()
	log.info('Initial stage complete.')


#This will check to see if the file uploaded is a CSV
def allowed_file(filename):
	return '.' in filename and \
	filename.rsplit('.',1)[1].lower() in ALLOWED_EXT # Splits the filename to check the extension of the file (csv)


#The root. It renders the login page
@app.route('/')
def home():
	log.info('Initial redirect')
	session['changedmodel'] = 'False'
	return redirect(url_for('index_train'))

#The training loading page
@app.route('/training_in_progress', methods=['GET', 'POST'])
def load_Train():
	#Global variables are needed
	global newReport, currentReport,temp_model
	temp_model = NN_model()
	if request.method == 'POST':
		log.info('Executing training sequence.')
		latest_report = ''
		try:
			#This is where training occurres.
			latest_report = temp_model.train(session['trainingFile'])
		except:
			#Error occurred
			errorString = 'Unable to generate a report based on the training data. Training data is invalid. Please upload another set of training data.'
			log.error(errorString)
			session['errorInfo'] = errorString
			session['errorOc'] = True
			return redirect(url_for('index_train'))

		if model is not None:
			previous_report = model.get_report()
		else:
			previous_report = None
		#If the data is not empty then theres a previous model, if it is empty then we display no model trained
		if previous_report == None:
			currentReport = [{"No Model has been trained before": "This is the first training of the model."}]
		else:
			currentReport = previous_report
		#Setting new report to show on UI
		newReport = latest_report
		session['fileChosen'] = session['trainingFile']
		app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
		log.info('Training complete.')
		return redirect(url_for('train_res'))
	else:
		return render_template('loading.html')


#This method will process the predictions
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	log.info('Currently on prediction page.')
	if request.method == 'POST' and request.files['file'] and allowed_file(request.files['file'].filename):
		log.info('File acquired.')
		file_to_process = request.files['file']
		filename = secure_filename(file_to_process.filename)
		file_to_process.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		log.info('File saved successfully.')
		#This will download the file
		return download(model.predict(file_to_process.filename))
	return render_template('index_Predict.html')

#This method will download the file from the predicted folder once the backend is finished
@app.route('/download', methods=['GET'])
def download(filename_1):
	log.info('Downloading processed file.')
	try:
		abs_path = app.root_path + DOWNLOAD_FOLDER
		log.info('Acquiring processed file from ' + abs_path)
		return send_from_directory(abs_path, filename=filename_1, as_attachment=True)
	except:
		errorString = 'Unknown error occurred.'
		log.error(errorString)
		return render_template('predict_submission.html', errorOcurred='True', errorInfo=errorString)
	return render_template('predict_submission.html')


#This method process the file for training
@app.route("/train", methods=['GET', 'POST'])
def index_train():
	log.info('Training process method called.')
	if session['changedmodel'] == 'True':
		session['changedmodel'] = 'False'
		return render_template('index_Train.html',modelChanged='True')
	elif request.method == 'POST':
		file_to_train = request.files['file']
		#This checks if the file exists and it is the correct format
		if file_to_train and allowed_file(file_to_train.filename):
			filename = secure_filename(file_to_train.filename)
			app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_TRAIN
			file_to_train.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			session['trainingFile'] = file_to_train.filename
			return redirect(url_for('load_Train'))
		elif request.form['change'] == 'Change Model':
			log.info('Model changed.')
			return redirect(url_for('change_model'))
	return render_template('index_Train.html')

@app.route("/change_model", methods=['GET','POST'])
def change_model():
	global model, temp_model, secondary_model
	log.info("-------------Model Changed-------------")
	session['changedmodel'] = 'True'
	if secondary_model is not None:
		temp = secondary_model
		secondary_model = model
		model = temp
	else:
		secondary_model = RR_model()
		temp = secondary_model
		secondary_model = model
		model = temp
	return redirect(url_for('index_train'))

#This is a loading screen and this also indexs the data to Mongo
@app.route('/loading', methods=['GET', 'POST'])
def load_accept(filename_load="nothing"):
	global temp_model,model
	if request.method != 'GET':
		filename_load = session['trainingFile']
	if request.method == 'GET':
		return render_template('accepted.html')
	elif request.method == 'POST' and filename_load != 'nothing':
		log.info('Executing saving sequence.')
		model = temp_model
		temp_model = None
		file_pickle = open(fname,'wb')
		pickled = pkl.dump(model,file_pickle)
		file_pickle.close()
		return redirect(url_for('index_train'))
	return render_template('accepted.html')


#This page displays the result of training
@app.route('/train_result', methods=['GET','POST'])
def train_res():
	global temp_model
	log.info('Training results rendered.')
	try:
		if request.method == 'POST':
			#These statements check to see the decision of the user
			if request.form['accept'] == 'Accept Changes':
				log.info('Changes accepted.')
				return redirect(url_for('load_accept'))
			elif request.form['accept'] == 'Discard Changes':
				#If the user discards the changes, this line will reset the temporary model
				temp_model = None
				log.info('Changes rejected. Discarding new model.')
				return redirect(url_for('index_train'))
			else:
				return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport)
	except Exception as e:
		log.error(type(e))# the exception instance
		log.error(e.args)      # arguments stored in .args
		log.error(e)           # __str__ allows args to be printed directly
		log.error(traceback.print_exception(e))
		errorString = 'Unable to gather model training results. Returning to training page.'
		log.error(errorString)
		session['errorOc'] = True
		session['errorInfo'] = errorString
		return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport, errorOcurred='True', errorInfo=errorString)
	return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport)





