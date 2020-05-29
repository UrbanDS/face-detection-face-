# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import json
import glob
import os
from skimage import io
import numpy as np
import dlib
import face_recognition
import socket
from PIL import Image, ImageFont, ImageDraw, ImageColor
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import networkx as nx
import math
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.patches as patches
import networkx as nx
import pandas as pd
import csv
import Twts_download as twts_dl
import Twts_img_download as img_dl
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "Wh-7LFCphhSSqkwbuMAPrDEcXRass7un"
secret = "qaoSkyLnpHVdBwhHWuaQGRnCavIqqhRS"
filepath = r"C:\Temp\1106767324818755585.jpg"
#image_url =r'https://pbs.twimg.com/ext_tw_video_thumb/1110391114991714305/pu/img/VN5DdIA3_jYhvxOw.jpg'
all_files = glob.glob(r'E:\Tweets\NoGEO\t2\*.jpg')
all_files = glob.glob(r'E:\Tweets\NoGEO\test\Images\20190327\*.jpg')
#  https://pbs.twimg.com/ext_tw_video_thumb/1110391114991714305/pu/img/VN5DdIA3_jYhvxOw.jpg;
#minimum_size = 1600 # pixel   besides profile
minimum_area = 1600 # pixel  besides profile
minimum_dimenssion = 400 # pixel   besides profile
def face2csv(filepath, face):

    face_list = []
    file_name = os.path.basename(filepath).replace('.jpg', '')
    image = Image.open(filepath)
    img_w = image.size[0]
    img_h = image.size[1]
    img_pixels = image.size[0] * image.size[1]
    del image
    try:
        attributes = face['attributes']
    except Exception as e:
        print('Error in getting face attributes: ', str(e), face)
        return None
    try:
        tid = file_name.split('-')[0]
        face_list.append(tid)

        try:
            img_index = file_name.split('-')[1]
        except:
            img_index = 1
        face_list.append(img_index)
        #

        face_token = face['face_token']
        face_list.append(face_token)

        ethnicity = face['attributes']['ethnicity']['value']
        face_list.append(ethnicity)

        gender = face['attributes']['gender']['value']
        face_list.append(gender)

        age = face['attributes']['age']['value']
        face_list.append(age)

        glass = face['attributes']['glass']['value']
        face_list.append(glass)

        smile = face['attributes']['smile']['value']
        face_list.append(smile)

        # e:emotion
        e_sadness = face['attributes']['emotion']['sadness']
        e_neutral = face['attributes']['emotion']['neutral']
        e_disgust = face['attributes']['emotion']['disgust']
        e_anger = face['attributes']['emotion']['anger']
        e_surprise = face['attributes']['emotion']['surprise']
        e_fear = face['attributes']['emotion']['fear']
        e_happiness = face['attributes']['emotion']['happiness']
        face_list.append(e_sadness)
        face_list.append(e_neutral)
        face_list.append(e_disgust)
        face_list.append(e_anger)
        face_list.append(e_surprise)
        face_list.append(e_fear)
        face_list.append(e_happiness)

        # b: beauty
        b_female = face['attributes']['beauty']['female_score']
        b_male = face['attributes']['beauty']['male_score']
        face_list.append(b_female)
        face_list.append(b_male)

        # m: mouthstatus
        m_close = face['attributes']['mouthstatus']['close']
        m_mask = face['attributes']['mouthstatus']['surgical_mask_or_respirator']
        m_open = face['attributes']['mouthstatus']['open']
        m_other = face['attributes']['mouthstatus']['other_occlusion']
        face_list.append(m_close)
        face_list.append(m_mask)
        face_list.append(m_open)
        face_list.append(m_other)

        # s: skinstatus
        s_dark = face['attributes']['skinstatus']['dark_circle']
        s_stain = face['attributes']['skinstatus']['stain']
        s_acne = face['attributes']['skinstatus']['acne']
        s_health = face['attributes']['skinstatus']['health']
        face_list.append(s_dark)
        face_list.append(s_stain)
        face_list.append(s_acne)
        face_list.append(s_health)

        # h: headpose
        h_yaw = face['attributes']['headpose']['yaw_angle']
        h_pitch = face['attributes']['headpose']['pitch_angle']
        h_roll = face['attributes']['headpose']['roll_angle']
        face_list.append(h_yaw)
        face_list.append(h_pitch)
        face_list.append(h_roll)

        face_quality = face['attributes']['facequality']['value']
        face_list.append(face_quality)

        # r: face rectangle
        r_width = face['face_rectangle']['width']
        r_top = face['face_rectangle']['top']
        r_left = face['face_rectangle']['left']
        r_height = face['face_rectangle']['height']
        face_list.append(r_width)
        face_list.append(r_top)
        face_list.append(r_left)
        face_list.append(r_height)

        # now it is area (pixel), will be changed outside this function.
        area_pixel = r_width * r_height
        face_list.append(area_pixel)

        try:
            eyegaze = face['eyegaze']
            print('Face detection, eyegaze: ', eyegaze)
        except:
            pass


        try:
            blur = face['attributes']['blur']['blurness']['value']
            #print('Face detection, blur: ', blur)
        except:
            blur = 0
            pass
        face_list.append(blur)

        portion = round(area_pixel / img_pixels, 3)

        face_list.append(portion)

    except Exception as e:
        print('Face detection result error：', str(e))
        print('')
        return None

    face_str = ''
    for item in face_list:
        face_str = face_str + str(item) + ','
    #print('face_str: ', face_str[:-1])

    return face_str[:-1]


def getFace(filepath, isDrawDox=True):
    isDrawBox = True
    image = Image.open(filepath)
    # if image.size[0] < minimum_size or image.size[1] < minimum_size:
    #     print("Face size less than {} size. ".format(minimum_area))
    #     return None
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())

    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')
    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    data.clear()

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=20)
        # get response
        qrcont = resp.read()

        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))

        faces = json.loads(qrcont.decode('utf-8'))['faces']
        #print('len of faces before faces[:5]: ', len(faces))
        faces = faces[:5] # Free API only return 5 face.
        #print('len of faces: ', len(faces), filepath)
        face_csv = ''
        if len(faces) == 0:
            return None
            #print('faces:', faces)
        # if len(faces) == 0:
        #     return None
        #     #print('faces:', faces)



        else:
            i = 0
            if isDrawBox:

                font = ImageFont.truetype(font=r'D:\YOLO\keras-yolo3-master\font\FiraMono-Medium.otf',
                                          size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

                thickness = (image.size[0] + image.size[1]) // 300
                draw = ImageDraw.Draw(image)
                image1 = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(image1)




            for face in faces:
                i += 1
                # if i > 4:
                #     #print('face index:  ', i)
                #     continue
                result = face2csv(filepath, face)
                if result is not None:
                    face_csv = face_csv + result + '\n'
                #print(face_list)
                # print(face['attributes']['ethnicity']['value'], face['attributes']['gender']['value'], face['attributes']['age']['value'])
                # print(face['face_rectangle'])

                if isDrawBox:
                #for i, c in reversed(list(enumerate(out_classes))):

                    ethnicity = face['attributes']['ethnicity']['value']
                    gender = face['attributes']['gender']['value']
                    age = face['attributes']['age']['value']

                    label = '{} {} {}'.format(ethnicity, gender, age)
                    #draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    #
                    #top, left, bottom, right = box
                    top = max(0, face['face_rectangle']['top'])
                    left = max(0, face['face_rectangle']['left'])
                    bottom = face['face_rectangle']['top'] + face['face_rectangle']['height']
                    right = face['face_rectangle']['left'] + face['face_rectangle']['width']

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])
                    color = getColor(ethnicity)
                    # My kingdom for a good redistributable image drawing library.

                    # draw rectangle for the face_recognization lib

                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=color)

                    # if df.ix[i, 'top'] - label_size[1] >= 0:
                    #     text_origin = np.array([df.ix[i, 'left'], df.ix[i, 'top'] - label_size[1]])
                    # else:
                    #     text_origin = np.array([df.ix[i, 'left'], df.ix[i, 'top'] + 1])
                    #
                    # # My kingdom for a good redistributable image drawing library.
                    # #print(label, (left, top), (right, bottom))
                    # for i in range(thickness):
                    #     draw.rectangle(
                    #         [df.ix[i, 'left'] + i, df.ix[i, 'top'] + i, df.ix[i, 'right'] - i, df.ix[i, 'bottom'] - i],
                    #         outline=self.colors[c])

                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=color)
                    if gender == 'Female':
                        color = (255,20,147)
                    else:
                        color = (0,191,255)
                    draw.text(text_origin, label, fill=color, font=font)

            saved_path = os.path.join(os.path.dirname(filepath), 'Face')
            #isDrawBox = False
            # for face in face_locations:
            #
            #     if isDrawBox:
            #
            #         top = max(0, face[0])
            #         left = max(0, face[3])
            #         bottom = face[2]
            #         right = face[1]

                    # for i in range(thickness):
                    #     draw.rectangle(
                    #         [left + i, top + i, right - i, bottom - i],
                    #         outline=(255, 0, 0))

            if isDrawBox:
                if not os.path.exists(saved_path):
                    os.mkdir(saved_path)
                saved_filepath = os.path.join(saved_path, os.path.basename(filepath))

                image.save(saved_filepath)
                del draw
            return face_csv
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
    except socket.timeout as e:
        print('Face detection Error in: ',  type(e))    #catched

def getColor(ethnicity):
    #print('ethnicity in getColor():', ethnicity)
    ethnicity_color ={
        'WHITE': (255,255,255),
        'BLACK': (204,204,0),
        'ASIAN': (255,228,181),
        'INDIA': (210,105,30)}
    try:
        color = ethnicity_color[ethnicity]
    except:
        color = '(255,0,0)'
    return color

def getFacefrom_list(file_list):
    csv_header = r'tid,img_index,face_token,ethnicity,gender,age,glass,smile,e_sadness,e_neutral,e_disgust,e_anger,e_surprise,e_fear,e_happiness,b_female,b_male,m_close,m_mask,m_open,m_other,s_dark,s_stain,s_acne,s_health,h_yaw,h_pitch,h_roll,face_quality,r_width,r_top,r_left,r_height,area_pixel,blurness,portion'
    csv_header = csv_header + '\n'
    all_files = file_list
    for filepath in all_files:
        #print(filepath)
        try:
            result = getFace(filepath)
            if result is not None:
                csv_header = csv_header + result
        except Exception as e:
            print('Face detection error in getFacefrom_list(): ', str(e), filepath)
            print('')
            continue

    return csv_header

def EqualSplit(length, n):  # return a list of [start_index, end_index] . Has bug hare.
    n = math.ceil(n)
    if n > length:
        print("Error in EqualSplit(): Length is shorter than n! ")
        return
    if n == 0:
        n = 1
    part_len = length / n
    # length = (length)
    start = []
    # lst.append(0)
    for i in range(n):
        start.append(int(round(part_len * i, 0)))
    # lst.append(len(df) - 1)
    # start_ends = []
    #print(start)
    return start

def local_face_recog_1_process(all_files, results):
    # results: dictionary
    #print(all_files)
    #PID = os.getpid()
    #time0 = time.time()
    for idx, filepath in enumerate(all_files):
        try:
            img = io.imread(filepath)
            shape = img.shape
            if shape[0] < minimum_dimenssion or shape[1] < minimum_dimenssion:
                if "Profile" not in filepath:
                    continue
        # print(filepath)

            image = face_recognition.load_image_file(filepath)
            img_name = os.path.basename(filepath)
            face_location0 = face_recognition.face_locations(image)
            face_locations = []
            for location in face_location0:
                top, right, bottom, left = location
                area = (bottom - top) * (right - left)
                if area > minimum_area:
                    face_locations.append(location)
                elif "Profile" in filepath:
                    face_locations.append(location)
            #encodings = face_recognition.face_encodings(image, face_locations)
            #face_locations = face_recognition.face_locations(image, model='cnn')

            face_cnt = len(face_locations)
            if face_cnt > 0:
                encodings = face_recognition.face_encodings(image, face_locations)
                results[filepath] = {'locations': face_locations, 'encodings': encodings}
                finished = "{} / {}".format(idx, len(all_files))
                #msg = f"PID {PID:6d} found {face_cnt:2d} face(s) in: {img_name:>22}, has finished {finished}."
                #print(msg)
            #face_locations_cnn = face_recognition.face_locations(image, model='cnn')
            #encodings_cnn = face_recognition.face_encodings(image, face_locations_cnn)

            # face_recognition.compare_faces()
            # i = 0
            # for encoding in encodings:
            #     #print(encoding.tolist())
            #     #print(i)
            #     #print(encodings_cnn[0].tolist())
            #     #results[filepath] = encoding
            #     #print('Compared result: ', face_recognition.compare_faces([encoding], encodings_cnn[0]))
            #     i += 1
                # print(encoding.size)

        except Exception as e:
            print('Error in local_face_recog_1_process() ', str(e), filepath)
            continue
    #print("results:", results)
    #time1 = time.time()

    return results

def getEncoding_mp(list_photos, Process_Cnt):
    starts = EqualSplit(len(all_files), Process_Cnt)
    i = 0
    pools = Pool(processes=Process_Cnt)
    results = multiprocessing.Manager().dict()
    frames = []
    for i in range(Process_Cnt):
        if i < Process_Cnt-1:

            pools.apply_async(local_face_recog_1_process, args=(list_photos[starts[i]:starts[i+1]], results))
        else:
            pools.apply_async(local_face_recog_1_process, args=(list_photos[starts[i]:], results))
    pools.close()
    pools.join()

    pools.terminate()

    print('len of results:', len(results))

    #print(results)

    return results

def encoding_face_folder(img_path=None, Process_Cnt=8):
    all_files = glob.glob(os.path.join(img_path, r'*.jpg'))
    save_file = os.path.join(img_path, os.path.basename(img_path) + '.fcs')
    results = {}
    time0 = time.time()
    if len(all_files) < 1:
        print("No photos in: {}", img_path)
        with open(save_file, 'wb') as f:
            f.write(pickle.dumps(results.copy()))
        return
    #msg = f'Found {}'

    #print("Found {} images in folder {}.".format(str(len(all_files)), img_path))
    #Process_Cnt = 8
    Process_Cnt = min(Process_Cnt, len(all_files))

    save_file = os.path.join(img_path, os.path.basename(img_path) + '.fcs')
    #print("Result will be stored in: ", save_file)

    #results = getEncoding_mp(all_files, Process_Cnt)

    starts = EqualSplit(len(all_files), Process_Cnt)

    if Process_Cnt == 1:
        local_face_recog_1_process(all_files, results)

    else:
        results = multiprocessing.Manager().dict()
        pools = Pool(processes=Process_Cnt)
        for i in range(Process_Cnt):
            if i < Process_Cnt-1:
                pools.apply_async(local_face_recog_1_process, args=(all_files[starts[i]:starts[i+1]], results))
            else:
                pools.apply_async(local_face_recog_1_process, args=(all_files[starts[i]:], results))
        pools.close()
        pools.join()
        pools.terminate()
    time1 = time.time()
    print('PID:{} used {:0.1f} second to find {} faces in folder: {}, stored in {}'.format(os.getpid(), time1 - time0, len(results), img_path, save_file))

    with open(save_file, 'wb') as f:
        f.write(pickle.dumps(results.copy()))

def face_compare(start, end, faces, results, threshold=0.6):
    #print(faces.keys())
    try:

        images = list(faces.keys())
        images.sort(reverse=False)
        #print(images[start:end])
        tasks = images[start:end]
        #print(start, end)
        task_cnt = end - start
        for i in range(start, end):  # compare photos
            #print(faces[images[i]]['encodings'])
           # print("PID: {:6d} is processing {} / {}: {} .".format(os.getpid(), i - start + 1, task_cnt, images[i]))
            for j in range(i + 1, len(images)):
                img1_name = os.path.basename(images[i]).split('.')[0]
                img2_name = os.path.basename(images[j]).split('.')[0]
                # img1 = mpimg.imread(images[i])
                # img2 = mpimg.imread(images[j])
                # plt.close('all')
                # plt.figure(figsize=(15, 10))  # 设置窗口大小
                #
                # ax1 = plt.subplot(1, 2, 1)
                # ax1.imshow(img1)
                #
                # plt.title(str(i) + '  ' + img1_name)
                #
                # ax2 = plt.subplot(1, 2, 2)
                # ax2.imshow(img2)
                #
                # plt.title(str(j) + '  ' + img2_name)

                img2_matched_cnt = 0
                encoding1_idx = 0
                for encoding1, location1 in zip(faces[images[i]]['encodings'], faces[images[i]]['locations']):
                    encoding2_idx = 0
                    for encoding2, location2 in zip(faces[images[j]]['encodings'], faces[images[j]]['locations']):
                        cmp = (face_recognition.compare_faces([encoding1], encoding2, threshold))
                        #threshold:",threshold)

                        if cmp[0]:
                            img2_matched_cnt = img2_matched_cnt + 1

                            #print("PID: ", os.getpid(), 'found', img1_name, 'has', img2_matched_cnt, 'faces in', img2_name)

                            # top, right, bottom, left = location1
                            # rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r',
                            #                          facecolor='none')
                            # ax1.add_patch(rect)
                            # top, right, bottom, left = location2
                            # rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r',
                            #                          facecolor='none')
                            # ax2.add_patch(rect)

                            results.append((str(img1_name), str(encoding1_idx), str(img2_name), str((encoding2_idx))))
                        encoding2_idx += 1
                    encoding1_idx += 1

                # if img2_matched_cnt > 0:
                #     #pass
                #     # if img2_matched_cnt > 1:
                #     # plt.title(str(i) + " " + str(j))
                #     #plt.show()
                #     plt.show(block=False)
                #     plt.pause(3)
                #     plt.close('all')


    except Exception as e:
        print("Error in face_compare: ", str(e))

    return results

def faces_compare(faces_file, Process_Cnt=8, threshold=0.35):

    #print(faces_file)
    results = []
    faces = pickle.loads(open(faces_file, 'rb').read())
    images = list(faces.keys())
    if len(images) == 0:
        print('No images in: {}'.format(faces_file))
        return results
    #print(images)
    Process_Cnt = min(Process_Cnt, len(images))
    starts = EqualSplit(len(images), Process_Cnt)
    #print(starts)
    if Process_Cnt == 1:
        face_compare(faces, )

    else:
        results = multiprocessing.Manager().list()
        pools = Pool(processes=Process_Cnt)
        for i in range(Process_Cnt):
            if i < Process_Cnt-1:
                pools.apply_async(face_compare, args=(starts[i], starts[i + 1], faces, results, threshold))
            else:
                pools.apply_async(face_compare, args=(starts[i], len(images), faces, results, threshold))
        pools.close()
        pools.join()
        pools.terminate()
    #print('len of matches:', len(results))
    return results


def face_compare_single_process(user_folders, threshold=0.6):
    if not isinstance(user_folders, list):
        user_folders = [user_folders]
    try:
        for idx, img_path in enumerate(user_folders):
            try:
                #img_path = os.path.join(root_folder, img_path)
                time0 = time.time()
                #print("PID: {} is processing faces in folder: # {}  {}".format(os.getpid(), idx, img_path))
                faces_path = os.path.join(img_path, os.path.basename(img_path) + '.fcs')
                mth_path = os.path.join(img_path, os.path.basename(img_path) + '.mth')
                # if not os.path.exists(faces_path):
                #     continue
                all_files = glob.glob(os.path.join(img_path, '*.jpg'))
                faces = {}
                if len(all_files) < 1:
                    print("No photos in: {}", img_path)
                    with open(faces_path, 'wb') as f:
                        f.write(pickle.dumps(faces.copy()))
                    continue
                time1 = time.time()
                faces = local_face_recog_1_process(all_files, faces)
                time2 = time.time()
                print("PID: {} used {:0.0f} seconds to find faces.".format(os.getpid(), time2 - time1))
                if not faces:
                    print("No face in: ", img_path)
                    continue

                with open(faces_path, 'wb') as f:
                    f.write(pickle.dumps(faces.copy()))

                images = list(faces.keys())
                matches = []
                matches = face_compare(0, len(images), faces, matches, threshold)
                with open(mth_path, 'w') as w:
                    for line in matches:
                        # print(','.join(line))
                        w.writelines(','.join(line) + '\n')
                face_attr = getUserFace_From_Matches(mth_path)
                time3 = time.time()
                print("PID: {} used {:0.0f} second to find user's face.".format(os.getpid(), time3 - time2))
                print(face_attr)

            except Exception as ee:
                print('Error in face_compare_single_process() loop: ', str(ee))
                continue

    except Exception as e:
        print("Error in face_compare_single_process: ", str(e))

def getArea(location):
    top, right, bottom, left = location
    w = right - left
    h = bottom - top
    return w * h

def getUserFace_From_Matches(mth_path):
    faces_path = mth_path[:-3] + 'fcs'
    userid = os.path.basename(mth_path).replace('.mth', '')


    faces = pickle.loads(open(faces_path, 'rb').read())
    df = pd.read_csv(mth_path, names=('img1', 'idx1', 'img2', 'idx2'))

    profile_path = os.path.join(os.path.dirname(faces_path), 'Profile_' + userid + '.jpg')
    img_path = profile_path
    best = img_path
    #print('img_path: ', img_path)
    profile_face = faces.get(profile_path, '')
    #print("Profile_face: ", profile_face)

    if len(df) < 1 and faces.get(profile_path, True):
        print("No match in ", mth_path)
        if profile_face == '':
            print('No face in the profile: ', mth_path)
            return

    df['node1'] = df['img1'].astype(str) + '_' + df['idx1'].astype(str)
    df['node2'] = df['img2'].astype(str) + '_' + df['idx2'].astype(str)

    #print(len(df))

    if len(df) > 0:

        G = nx.Graph()
        for idx, row in df.iterrows():
            G.add_edge(row['node1'], row['node2'])
        cc = nx.algorithms.connected_components(G)
        lens = [len(cc) for cc in sorted(nx.connected_components(G), key=len, reverse=True)]
        largest_cc = max(nx.connected_components(G), key=len)
        cc = sorted(nx.connected_components(G), key=len, reverse=True)

        degrees = dict(G.degree())

        c = cc[0]  # keep the largest one..
        # print("cc: ", cc)
        folder = os.path.dirname(mth_path)
        list_c = list(c)
        #print('c: ', c)

        c_degrees = [degrees[x] for x in list_c]
        indecies = list(map(lambda node: int(node.split('_')[-1]), list_c))
        images = list(map(lambda node: os.path.join(folder, '_'.join(node.split('_')[0:-1]) + '.jpg'), list_c))
        locs = list(map(lambda img_path, idx: faces[img_path]['locations'][idx], images, indecies))
        c_areas = list(map(getArea, locs))

        largest_area_idx = c_areas.index(max(c_areas))
        c_locations = [faces[x]['locations'] for x in images]

        max_degree_idx = c_degrees.index(max(c_degrees))
        best = list_c[max_degree_idx]
        largest = list_c[largest_area_idx]


        img_path = os.path.join(folder, best.split('_')[0] + '.jpg')
        if "Profile_" in best:
            # print(node)
            # print('_'.join(node.split('_')[:-1]))
            img_path = os.path.join(folder, best + '.jpg')
        #print('img_path: ', img_path)

    locations = faces[img_path]['locations']
    try:
        face_idx = int(best.split('_')[-1])
    except:    # the profile
        c_areas = list(map(getArea, locations))
        largest_area_idx = c_areas.index(max(c_areas))
        face_idx = largest_area_idx

    top, right, bottom, left = locations[face_idx]
    w = right - left
    h = bottom - top
    r = 0.4

    img = mpimg.imread(img_path)
    shape = img.shape

    ex_top = int(max(0, top - h * r))
    ex_bottom = int(min(shape[0], bottom + h * r))
    ex_right = int(min(shape[1], right + w * r))
    ex_left = int(max(0, left - w * r))

    cropped = img[ex_top:ex_bottom, ex_left:ex_right]

    cropped_file = os.path.join(os.path.dirname(img_path), userid + '_face.png')
    #print('img_path:', img_path)
    #print('Cropped user face file: ', cropped_file)
    mpimg.imsave(cropped_file, cropped)

    face_attr = getFacefrom_list([cropped_file])
    #print(face_attr)
    face_attr_file = os.path.join(os.path.dirname(img_path), userid + '_face.csv')
    with open(face_attr_file, 'w') as f:
        f.writelines(face_attr)
    return face_attr





        # img_path = os.path.join(folder, largest.split('_')[0] + '.jpg')
        # if largest.split('_')[0] + '.jpg' == "Profile.jpg":
        #     # print(node)
        #     # print('_'.join(node.split('_')[:-1]))
        #     img_path = os.path.join(folder, '_'.join(largest.split('_')[:-1]) + '.jpg')
        #
        # locations = faces[img_path]['locations']
        # face_idx = int(largest.split('_')[-1])
        #
        # top, right, bottom, left = locations[face_idx]
        # w = right - left
        # h = bottom - top
        # rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r', facecolor='none')
        #
        #
        # for idx, node in enumerate(c):
        #     img_path = os.path.join(folder, node.split('_')[0] + '.jpg')
        #     if node.split('_')[0] + '.jpg' == "Profile.jpg":
        #         # print(node)
        #         # print('_'.join(node.split('_')[:-1]))
        #         img_path = os.path.join(folder, '_'.join(node.split('_')[:-1]) + '.jpg')
        #
        #     locations = faces[img_path]['locations']
        #     face_idx = int(node.split('_')[-1])
        #
        #
        #     top, right, bottom, left = locations[face_idx]
        #     w = right - left
        #     h = bottom - top
        #     r = 0.2
        #
        #     shape = img.shape
        #
        #     ex_top = int(max(0, top - h * r))
        #     ex_bottom = int(min(shape[0], bottom + h * r))
        #     ex_right = int(min(shape[1], right + w * r))
        #     ex_left = int(max(0, left - w * r))
        #
        #     cropped = img[ex_top:ex_bottom, ex_left:ex_right]

def getUserFaceAttrFromIds(ids, face_attrs, saved_path):
    # face_attrs: a shared list
    # ids: a list or a single user id

    try:
        start = time.time()
        if not isinstance(ids, list):
            ids = [ids]
        twts_dl.auth_tweepy_api()
        print('len(ids): ', len(ids))
        for idx, id in enumerate(ids):
            try:
                time0 = time.time()
                username = ''
                try:
                    user = twts_dl.tweepy_api.get_user(id)  # will consume rate
                    username = user.screen_name
                except Exception as eee:
                    print("Error in get user name: ", id, str(eee))
                    pass

                # twts_csv = twts_dl.getUsersTwts([(username, id)])
                # df_tweets = pd.read_csv(StringIO(twts_csv), quoting=csv.QUOTE_NONE, dtype={'tweetID': str},
                #                         delimiter=',', error_bad_lines=False, low_memory=False)
                # if len(df_tweets) == 0:
                #     continue
                # username = df_tweets.iloc[0]['username']
                user_folder = os.path.join(saved_path, id + "_" + username)
                img_dl.getUserImages(username, id, saved_path)
                #encoding_face_folder(user_folder, Process_Cnt=1)
                #faces_path = os.path.join(user_folder, id + "_" + username + '.fcs')
                face_compare_single_process(user_folder, threshold=0.45)
                # print('faces_path: ', faces_path)  # good
                #results = faces_compare(faces_path, Process_Cnt=1, threshold=0.45)
                #     with open(mth_path, 'w') as w:
                #         for line in results:
                #             #print(','.join(line))
                #             w.writelines(','.join(line) + '\n')
                # if not os.path.exists(folder):
                #     os.mkdir(folder)
                #
                # csv_path = os.path.join(saved_path, id + "_" + username, id + "_" + username + '.csv')
                # with open(csv_path, 'w', encoding='utf-8') as f:
                #     f.writelines(twts_csv)
                #print('username: ', results)
                #print(len(df_tweets))
                time1 = time.time()
                time_cost = timedelta(seconds=int(time.time() - start))
                print(
                    "PID: {} used {:0.1f} seconds in total to finish user {}, id:{}. Processed {} users in {}\n".format(os.getpid(), time1 - time0, username, id, idx + 1, time_cost))

            except Exception as ee:
                print("Error in process user id: ", id, str(ee))
                continue
    except Exception as e:
        print("Error in getUserFaceAttrFromIds():", str(e))

    #print("PID: {} used {:0.1f} seconds in total to finish {} folders.".format(os.getpid(), time1 - time0, ))
    return face_attrs


def getUserFaceAttrFromFile(ids_csv, Process_Cnt=1, p_offset=0):   # input the ids list csv file, p_offset is the offset of each process, not the file.
    df = pd.read_csv(ids_csv, dtype=str, names=['userid'], header=None)
    #print(df[:10])
    user_ids = df['userid'].tolist()
    saved_path = os.path.dirname(ids_csv)
    starts = EqualSplit(len(df), Process_Cnt)
    pools = Pool(processes=Process_Cnt)
    face_attrs = multiprocessing.Manager().list()
    for i in range(Process_Cnt):
        try:
            if i < Process_Cnt - 1:
                # print(i, i + 1)
                # print(starts[i], starts[i + 1])
                inputs = user_ids[starts[i] + p_offset:starts[i + 1]]
                pools.apply_async(getUserFaceAttrFromIds, args=(inputs, face_attrs, saved_path))
            else:
                inputs = user_ids[starts[i] + p_offset:]
                pools.apply_async(getUserFaceAttrFromIds, args=(inputs, face_attrs, saved_path))
            #print(inputs)
        except Exception as e:
            print("Error in pools.apply_async: ", str(e))
    pools.close()
    pools.join()
    pools.terminate()
    print("Finished! ")



if __name__ == '__main__':

    getUserFaceAttrFromFile(r'F:\Tweets\User20190420_images\userID20190420.csv', Process_Cnt=10)
    #all_files = glob.glob(r'F:\Tweets\images\Flood_users\profiles\*.png')

    # getUserFace_From_Matches(r'F:\Tweets\images\user2018\203613767_Eric_Lee_822\203613767_Eric_Lee_822.mth')
    #
    # img_path = r'F:\Tweets\images\users\3127881897_lalalahannahh'

    # root_folder = r'F:\Tweets\images\user2018'
    # threshold = 0.45
    # user_folders = os.listdir(root_folder)
    # user_folders = [os.path.join(root_folder, folder) for folder in user_folders]
    # #print(user_folders)
    # #user_folders = user_folders[200:]
    # user_folders = user_folders[590:]
    # print(len(user_folders))
    # #user_folders = [r'F:\Tweets\images\users\17983161_tweedledeedumm1']
    #
    # Process_Cnt = 5
    # starts = EqualSplit(len(user_folders), Process_Cnt)
    # print(starts)
    # pools = Pool(processes=Process_Cnt)
    # results = multiprocessing.Manager().list()
    # for i in range(Process_Cnt):
    #     if i < Process_Cnt-1:
    #         pools.apply_async(face_compare_single_process, args=(user_folders[starts[i]:starts[i + 1]], results, threshold))
    #     else:
    #         pools.apply_async(face_compare_single_process, args=(user_folders[starts[i]:len(user_folders)], results, threshold))
    # pools.close()
    # pools.join()
    # pools.terminate()

    ##########################################



    ##############################
    #
    # for idx, img_path in enumerate(user_folders):
    #     img_path = os.path.join(root_folder, img_path)
    #     print("Processing folder: #", idx, img_path)
    #     faces_path = os.path.join(img_path, os.path.basename(img_path) + '.fcs')
    #     mth_path = os.path.join(img_path, os.path.basename(img_path) + '.mth')
    #     # if not os.path.exists(faces_path):
    #     #     continue
    #     encoding_face_folder(img_path, 8)
    #     results = faces_compare(faces_path, 8, threshold)
    #     with open(mth_path, 'w') as w:
    #         for line in results:
    #             #print(','.join(line))
    #             w.writelines(','.join(line) + '\n')
    #
    # ####################################################################################
    # faces = pickle.loads(open(faces_path, 'rb').read())
    # results = pd.read_csv(mth_path)
    # df = pd.read_csv(mth_path, names=('img1', 'idx1', 'img2', 'idx2'))
    # df['node1'] = df['img1'].astype(str) + '_' + df['idx1'].astype(str)
    # df['node2'] = df['img2'].astype(str) + '_' + df['idx2'].astype(str)
    # G = nx.Graph()
    # for idx, row in df.iterrows():
    #     G.add_edge(row['node1'], row['node2'])
    # cc = nx.algorithms.connected_components(G)
    # lens = [len(cc) for cc in sorted(nx.connected_components(G), key=len, reverse=True)]
    # largest_cc = max(nx.connected_components(G), key=len)
    # cc = sorted(nx.connected_components(G), key=len, reverse=True)
    #
    # for c in cc:
    #     col = math.ceil(math.sqrt(len(c)))
    #     row = math.ceil(len(c)/col)
    #     folder = os.path.dirname(mth_path)
    #     fig, ax = plt.subplots(nrows=row, ncols=col, squeeze=True)
    #     plt.figure(figsize=(15, 10))  # 设置窗口大小
    #
    #     for idx, node in enumerate(c):
    #         ax = plt.subplot(row, col, idx + 1)
    #         img_path = os.path.join(folder, node.split('_')[0] + '.jpg')
    #         if node.split('_')[0] + '.jpg' == "Profile.jpg":
    #             print(node)
    #             print('_'.join(node.split('_')[:-1]))
    #             img_path = os.path.join(folder, '_'.join(node.split('_')[:-1]) + '.jpg')
    #         img = mpimg.imread(img_path)
    #         locations = faces[img_path]['locations']
    #         face_idx = int(node.split('_')[-1])
    #         ax.imshow(img)
    #         top, right, bottom, left = locations[face_idx]
    #         rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r',
    #                                  facecolor='none')
    #         ax.add_patch(rect)
    #     plt.show()
    #     plt.close('all')


    # #print(cc)
    # print(largest_cc)
    # plt.subplots(12)
    # #plt.tight_layout()
    # for idx, c in enumerate(cc):
    #     print(c)

    #nx.draw(G, with_labels=True, fillcolor='cyan2')
    # plt.subplot(122)
    # nx.draw_shell(G, with_labels=True)
    #plt.show()



    #results = getFacefrom_list(all_files)
    #print(results)
    #f = open(r'F:\Tweets\images\Flood_users\profiles\Face\results.csv', 'w')
    #f.writelines(results)
    #f.close()
    #filepath = r'D:\spl499660_002.jpg'
    #print(getFace(filepath))
    #image = face_recognition.load_image_file(filepath)

    #face_locations = face_recognition.face_locations(image)
    #face_landmarks_list = face_recognition.face_landmarks(image)
    #print(face_locations)


