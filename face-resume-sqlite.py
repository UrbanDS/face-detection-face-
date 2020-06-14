#!/usr/bin/python
import argparse 
import urllib.request
import urllib.error
import time
import json
import socket
from datetime import datetime
import glob
import csv
import os

from database import mydatabase
dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='mydb.sqlite')

http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'

api_key = 'AFWkr-35C8LACVx1raEXqnJMwb494tkA'
api_secret = 'Rj1-hdcAQxmQKWRQgiZtcQO8VxKT30E0'

# Initialize parser 
parser = argparse.ArgumentParser() 
  
# Adding optional argument 
# parser.add_argument("-n", "--New", default=None, help = "create new table") 
parser.add_argument("-t", "--Table",default='face_attributes_atlanta', help = "access existed table") 
parser.add_argument("-c", "--Iscsv", default=False, help = "store data in csv or not") 
  
args = parser.parse_args()


def getFaceAttributes():
    columns = ['image_path', 'gender', 'age', 'smile_value', 'smile_threshold', 'emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_happiness', 'emotion_neutral', 'emotion_sadness', 'emotion_surprise', 'ethnicity', 'beauty_male_score', 'beauty_female_score', 'mouthstatus_surgical_mask_or_respirator','mouthstatus_other_occlusion', 'mouthstatus_close', 'mouthstatus_open', 'skinstatus_health', 'skinstatus_stain', 'skinstatus_dark_circle', 'skinstatus_acne', 'face_rectangle_top', 'face_rectangle_left', 'face_rectangle_width', 'face_rectangle_height', 'status', 'time_stamp']
    columns_str = ",".join(columns)
    i=0
    for filepath in glob.iglob(r'../atlanta/*.jpg'):
        file_name = os.path.basename(filepath)
        # check in database whether the file name is present or not
        file_check_query = "select count(*) from {} where image_path like '%{}%'".format(args.Table,file_name)
        # print(file_check_query)
        i+=1
        print(i)
        count = dbms.get_count_result(file_check_query)
        # print(count)
        if count >0 : continue
        print(i,"saving")
        rows, message = getFace(filepath)

        if rows is not None and message is 'success':
            dbms.insertmany_sqlite3(args.Table,columns_str,rows)
        elif rows is None :
        	print("none")
        	time_stamp=datetime.now()
        	rows = []
        	vals=["'"+filepath+"'", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL", "'"+message+"'", "'"+str(time_stamp)+"'"]
        	f_values = ",".join(val for val in vals)
        	rows.append(f_values)
        	dbms.insertmany_sqlite3(args.Table,columns_str,rows)

        if args.Iscsv:
            with open("face_attributes.csv", 'a') as csvfile:
                csvwriter = csv.writer(csvfile,  lineterminator='\n')
                csvwriter.writerows(rows)


def getFace(imagepath):
	boundary = '----------%s' % hex(int(time.time() * 1000))
	data = []
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
	data.append(api_key)
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
	data.append(api_secret)
	data.append('--%s' % boundary)
	fr = open(imagepath, 'rb')
	data.append(
		'Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
	data.append('Content-Type: %s\r\n' % 'application/octet-stream')
	data.append(fr.read())

	fr.close()
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' %
				'return_landmark')
	data.append('1')
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' %
				'return_attributes')
	data.append(
		"gender,smiling,age,emotion,ethnicity,beauty,mouthstatus,skinstatus")
	data.append('--%s--\r\n' % boundary)

	for i, d in enumerate(data):
		if isinstance(d, str):
			data[i] = d.encode('utf-8')
	http_body = b'\r\n'.join(data)

	# build http request
	req = urllib.request.Request(url=http_url, data=http_body)

	# header
	req.add_header(
		'Content-Type', 'multipart/form-data; boundary=%s' % boundary)
	data.clear()

	try:
		# post data to server
		resp = urllib.request.urlopen(req, timeout=20)
		# get response
		qrcont = resp.read()

		# if you want to load as json, you should decode first,
		# for example: json.loads(qrount.decode('utf-8'))

		faces = json.loads(qrcont.decode('utf-8'))['faces']
		if len(faces) == 0:
			return None, 'No Faces'
		else:
			i = 0
			all_data = []
			for face in faces:
				i += 1
				# print(face)
				if face is not None:
					result = faceSQLformat(face, imagepath)
					if result is not None:
						all_data.append(result)
					# face_csv = face_csv + result + '\n'
					# print(result)
					# return result
			return all_data, 'success'
	except urllib.error.HTTPError as e:
		# print("--------------")
		# print(e.read().decode('utf-8'))
		# print("-----------------")
		return None, e
	except socket.timeout as e:
		# print('Face detection Error in: ', type(e))  # catched
		return None, e


def faceSQLformat(faceAttri, image_path):
	if 'attributes' not in faceAttri:
		return None
	face = faceAttri['attributes']
	gender = face['gender']['value'] if bool(face['gender']['value']) else None
	age = face['age']['value'] if bool(face['age']['value']) else None
	smile_value = face['smile']['value'] if bool(
		face['smile']['value']) else None
	smile_threshold = face['smile']['threshold'] if bool(
		face['smile']['threshold']) else None
	emotion_anger = face['emotion']['anger'] if bool(
		face['emotion']['anger']) else None
	emotion_disgust = face['emotion']['disgust'] if bool(
		face['emotion']['disgust']) else None
	emotion_fear = face['emotion']['fear'] if bool(
		face['emotion']['fear']) else None
	emotion_happiness = face['emotion']['happiness'] if bool(
		face['emotion']['happiness']) else None
	emotion_neutral = face['emotion']['neutral'] if bool(
		face['emotion']['neutral']) else None
	emotion_sadness = face['emotion']['sadness'] if bool(
		face['emotion']['sadness']) else None
	emotion_surprise = face['emotion']['surprise'] if bool(
		face['emotion']['surprise']) else None
	ethnicity = face['ethnicity']['value'] if bool(face['ethnicity']['value']) else None
	beauty_male_score = face['beauty']['male_score'] if bool(
		face['beauty']['male_score']) else None
	beauty_female_score = face['beauty']['female_score'] if bool(
		face['beauty']['female_score']) else None
	mouthstatus_surgical_mask_or_respirator = face['mouthstatus']['surgical_mask_or_respirator'] if bool(
		face['mouthstatus']['surgical_mask_or_respirator']) else None
	mouthstatus_other_occlusion = face['mouthstatus']['other_occlusion'] if bool(
		face['mouthstatus']['other_occlusion']) else None
	mouthstatus_close = face['mouthstatus']['close'] if bool(
		face['mouthstatus']['close']) else None
	mouthstatus_open = face['mouthstatus']['open'] if bool(
		face['mouthstatus']['open']) else None
	skinstatus_health = face['skinstatus']['health'] if bool(
		face['skinstatus']['health']) else None
	skinstatus_stain = face['skinstatus']['stain'] if bool(
		face['skinstatus']['stain']) else None
	skinstatus_dark_circle = face['skinstatus']['dark_circle'] if bool(
		face['skinstatus']['dark_circle']) else None
	skinstatus_acne = face['skinstatus']['acne'] if bool(
		face['skinstatus']['acne']) else None
	face_rectangle_top = faceAttri['face_rectangle']['top'] if bool(
		faceAttri['face_rectangle']['top']) else None
	face_rectangle_left = faceAttri['face_rectangle']['left'] if bool(
		faceAttri['face_rectangle']['left']) else None
	face_rectangle_width = faceAttri['face_rectangle']['width'] if bool(
		faceAttri['face_rectangle']['width']) else None
	face_rectangle_height = faceAttri['face_rectangle']['height'] if bool(
		faceAttri['face_rectangle']['height']) else None
	status = 'processed'
	time_stamp = datetime.now()
	values = [image_path, gender, age, smile_value, smile_threshold, emotion_anger, emotion_disgust, emotion_fear,
				emotion_happiness, emotion_neutral, emotion_sadness, emotion_surprise, ethnicity, beauty_male_score,
				beauty_female_score, mouthstatus_surgical_mask_or_respirator, mouthstatus_other_occlusion,
				mouthstatus_close, mouthstatus_open, skinstatus_health, skinstatus_stain, skinstatus_dark_circle,
				skinstatus_acne, face_rectangle_top, face_rectangle_left, face_rectangle_width, face_rectangle_height, status, time_stamp]
	none_null = ["NULL" if val == None else "'"+str(val)+"'" for val in values]
	f_values = ",".join([str(val) for val in none_null])
	return f_values

getFaceAttributes()

