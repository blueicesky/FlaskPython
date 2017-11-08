from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from configparser import SafeConfigParser
from logger import Logger
import pandas as pd
import numpy as np
import csv
import time
import sys

config = SafeConfigParser()
log = Logger('nn_model.log')
config.read('config.ini')
sys.setrecursionlimit(10000)

hidden_1 = config.getint('NeuralNetConfig','HIDDEN_1')
hidden_2 = config.getint('NeuralNetConfig','HIDDEN_2')
hidden_3 = config.getint('NeuralNetConfig','HIDDEN_3')
hidden_4 = config.getint('NeuralNetConfig','HIDDEN_4')
rstate = config.getint('NeuralNetConfig','RSTATE')
activationFunc = config.get('NeuralNetConfig','ACTIVATION')
m_iter =config.getint('NeuralNetConfig','MAX_ITER')
l_rate = config.getfloat('NeuralNetConfig','LEARNING_RATE')
ver_bose = config.getboolean('NeuralNetConfig','VERBOSE')

t_size = config.getfloat('Misc','TEST_SIZE_1')
digit_num = config.getint('Misc','DIGIT_NUM')


class RR_model:
	def __init__(self):
		log.info('Initializing Model.')
		self.current_report = None
		log.info('Initializing Random Forest Classifier')
		self.model = RandomForestClassifier(n_estimators=200)

	def get_report(self):
		log.info('Getting current classification report.')
		return self.current_report

	def train(self,filename):
		csv_path = 'train_data/' + str(filename)
		df = pd.read_csv(csv_path)
		# Add split function to split target and features
		classification_report_var = None
		X = df.drop('Survived',axis=1)
		y = df['Survived']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size,random_state=101)
		self.model.fit(X_train,y_train)
		# Making predictions with X_test, the test data set.
		pred = self.model.predict(X_test)
		# print(classification_report(y_test,pred,digits=5))
		classification_report_var = classification_report(y_test,pred,digits=digit_num)

		log.info('Creating classification report in JSON.')
		classes = []
		lines = classification_report_var.split('\n')
		for line in lines[2:-3]:
			row = {}
			row_data = line.split()
			row['class'] = float(row_data[0])
			row['precision'] = float(row_data[1])
			row['recall'] = float(row_data[2])
			row['f1_score'] = float(row_data[3])
			row['support'] = float(row_data[4])
			classes.append(row)
		temp_row_data = lines[-2].split()
		temp_row  = {'precision': temp_row_data[3], 'recall': temp_row_data[4], 'f1_score': temp_row_data[5],
					'support': temp_row_data[6]}
		classes.append(temp_row)
		self.current_report = classes
		return self.current_report


	def predict(self,filename):
		log.info('Prediction sequence started.')
		csv_path = 'temp_predict/'+str(filename)
		df = pd.read_csv(csv_path)
		pred = self.model.predict(df)
		df_pred = pd.DataFrame(pred,columns=['Predictions'])
		final_frame = pd.concat([df_pred,df.drop('Unnamed: 0',axis=1)],axis=1)
		# set filename according to timestamp and return filename
		ts = time.time()
		ts = int(ts)
		ts = str(ts)
		filename_path = "temp_predicted/"+"predicted_" + ts + ".csv"
		filename = "predicted_" + ts + ".csv"
		final_frame.to_csv(filename_path)
		log.info('Prediction done.')
		return filename


