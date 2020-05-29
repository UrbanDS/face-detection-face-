from database import mydatabase
from facepplib import FacePP
import threading, time
import math

dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='mydb.sqlite')
# dbms.create_db_tables()

facepp = FacePP(api_key='g06BZoz_VFEJxvzaLPTlQI-ZsbhCbRbU',api_secret='EHoHJiGyJn-IoeHNBMGNlUFBKhoLmyci')

def get_face_Images():

	lop = math.ceil(3635/2)
	x = 1
	counter = 1
	for i in range(lop):
		# ----
		query = "SELECT * from imagelist where id Between {} and {};".format(x,counter*2)
		x = counter*2+1
		counter = counter+1
		imageList = dbms.get_all_data('', query)
		# imageList = dbms.get_all_data('imagelist','')
		# print(imageList)
		for img in imageList:
			# print(image)
			image = facepp.image.get(image_file=img[1],return_attributes=['gender', 'smiling', 'age', 'emotion', 'ethnicity', 'beauty','mouthstatus', 'skinstatus'])

			for i in range(len(image.faces)):
				# for face in faces:
				face = image.faces[i]
				image_id = img[0]
				gender = "'" + face.gender['value'] + "'" if bool(face.gender['value']) else None
				age = face.age['value'] if bool(face.age['value']) else None
				smile_value = face.smile['value'] if bool(face.smile['value']) else None
				smile_threshold = face.smile['threshold'] if bool(face.smile['threshold']) else None
				emotion_anger = face.emotion['anger'] if bool(face.emotion['anger']) else None
				emotion_disgust = face.emotion['disgust'] if bool(face.emotion['disgust']) else None
				emotion_fear = face.emotion['fear'] if bool(face.emotion['fear']) else None
				emotion_happiness = face.emotion['happiness'] if bool(face.emotion['happiness']) else None
				emotion_neutral = face.emotion['neutral'] if bool(face.emotion['neutral']) else None
				emotion_sadness = face.emotion['sadness'] if bool(face.emotion['sadness']) else None
				emotion_surprise = face.emotion['surprise'] if bool(face.emotion['surprise']) else None
				ethnicity = "'" + face.ethnicity['value'] + "'" if bool(face.ethnicity['value']) else None
				beauty_male_score = face.beauty['male_score'] if bool(face.beauty['male_score']) else None
				beauty_female_score = face.beauty['female_score'] if bool(face.beauty['female_score']) else None
				mouthstatus_surgical_mask_or_respirator = face.mouthstatus['surgical_mask_or_respirator'] if bool(face.mouthstatus['surgical_mask_or_respirator']) else None
				mouthstatus_other_occlusion = face.mouthstatus['other_occlusion'] if bool(face.mouthstatus['other_occlusion']) else None
				mouthstatus_close = face.mouthstatus['close'] if bool(face.mouthstatus['close']) else None
				mouthstatus_open = face.mouthstatus['open'] if bool(face.mouthstatus['open']) else None
				skinstatus_health = face.skinstatus['health'] if bool(face.skinstatus['health']) else None
				skinstatus_stain = face.skinstatus['stain'] if bool(face.skinstatus['stain']) else None
				skinstatus_dark_circle = face.skinstatus['dark_circle'] if bool(face.skinstatus['dark_circle']) else None
				skinstatus_acne = face.skinstatus['acne'] if bool(face.skinstatus['acne']) else None
				face_rectangle_top = face.face_rectangle['top'] if bool(face.face_rectangle['top']) else None
				face_rectangle_left = face.face_rectangle['left'] if bool(face.face_rectangle['left']) else None
				face_rectangle_width = face.face_rectangle['width'] if bool(face.face_rectangle['width']) else None
				face_rectangle_height = face.face_rectangle['height'] if bool(face.face_rectangle['height']) else None
				columns = ['image_id', 'gender', 'age', 'smile_value', 'smile_threshold', 'emotion_anger','emotion_disgust', 'emotion_fear', 'emotion_happiness', 'emotion_neutral', 'emotion_sadness','emotion_surprise', 'ethnicity', 'beauty_male_score', 'beauty_female_score','mouthstatus_surgical_mask_or_respirator', 'mouthstatus_other_occlusion','mouthstatus_close', 'mouthstatus_open', 'skinstatus_health', 'skinstatus_stain','skinstatus_dark_circle', 'skinstatus_acne', 'face_rectangle_top', 'face_rectangle_left','face_rectangle_width', 'face_rectangle_height']
				values = [image_id, gender, age, smile_value, smile_threshold, emotion_anger, emotion_disgust,emotion_fear, emotion_happiness, emotion_neutral, emotion_sadness, emotion_surprise,ethnicity, beauty_male_score, beauty_female_score, mouthstatus_surgical_mask_or_respirator,mouthstatus_other_occlusion, mouthstatus_close, mouthstatus_open, skinstatus_health,skinstatus_stain, skinstatus_dark_circle, skinstatus_acne, face_rectangle_top,face_rectangle_left, face_rectangle_width, face_rectangle_height]
				none_null = ["NULL" if val == None else val for val in values]
				f_values = ",".join([str(val) for val in none_null])
				dbms.insertmany_sqlite3("faceAttributes", ",".join(columns), [f_values])
		time.sleep(40)
	print("---- data insertion done")

	# print(image.faces[0].gender['value'])
	# print(image.faces[0].smile)
	# print(image.faces[0].age['value'])
	# print(image.faces[0].emotion)
	# print(image.faces[0].ethnicity)
	# print(image.faces[0].beauty)
	# print(image.faces[0].mouthstatus)
	# print(image.faces[0].skinstatus)
	# print(image.faces[0].face_rectangle)


get_face_Images()

