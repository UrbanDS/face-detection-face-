#import twts_download

import Twts_img_download as Img_dl
import time
import Twts_download as twts_dl
from datetime import datetime
from keras_yolo3_master import yolo_twts as yolo_lib
#import fcntl
from googletrans import Translator
import configparser
import os
import multiprocessing as mp
import PyTorchFloodNN as ptFldNN
from multiprocessing import Process, Value, Array
import Database_operation as db_op
import re
import pandas as pd
database_paras = object
import random
con = object
import FacePP as fpp
#configs = object
from PIL import Image
from googletrans.constants import LANGCODES, LANGUAGES
import emoji
image_folder = r'F:\Tweets\images'
class config():
    curpath = ''
    cfgpath = ''
    sections = ''
    def get_configs(self):
        self.curpath = os.path.dirname(os.path.realpath(__file__))
        self.cfgpath = os.path.join(self.curpath, "cfg.ini")
        print(self.cfgpath)  # cfg.ini的路径

        # 创建管理对象
        conf = configparser.ConfigParser()

        ini = conf.read(self.cfgpath, encoding="utf-8")  # python3


        self.sections = conf.sections()

        return conf

        #print(sections)  # 返回list.

        #items = conf.items('twitter_auth')s
        #print(items)  # list里面对象是元祖
Twts_package_used_time = datetime.now()
Twts_package_updated_time = datetime.now()
tweets_package = ''
twts_pkg_list = []
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#df_tweets = []

def twts_download(configs, df_tweets):
    print(configs.sections())
    twts_dl.setTwtsDownload(configs, df_tweets)
    print('setTwtsDownload finished.')
    print('twts_dl.BBOX:', twts_dl.BBOX)
    print('twts_dl.keywords:', twts_dl.keywords)
    twts_dl.set_auth(configs)
    twts_dl.streamData(df_tweets)

def Imgs_download(configs, df_tweets):
    #print(configs.sections())
    Img_dl.setImageDownload(configs)
    #Img_dl.Batch_img_download()
    Img_dl.Batch_img_download_Keep_Processes(df_tweets)

def joinPath(row):
    return os.path.join(row['paths'], row['dates'], row['tid'])

def translation(input_str, dest=None):

    translate = Translator()
    if dest is not None:
        result = translate.translate(input_str, dest=dest)
    else:
        result = translate.translate(input_str)
    #print(result.text)
    return result.text

def transTwts(configs, dest):
    print('transTwts() started. ')
    try:

        #global database_paras, con
        sql_twts = db_op.SQL_tweets()

        host = configs['database']['host'].replace('"', '')
        user = configs['database']['user'].replace('"', '')
        password = configs['database']['password'].replace('"', '')
        db = configs['database']['db'].replace('"', '')

        db_info_list = []
        db_info_list.append(host)
        db_info_list.append(user)
        db_info_list.append(password)
        db_info_list.append(db)

        db_info_str = ','.join(db_info_list)

        database_paras = db_op.Database_parameters(host, user, password, db)


        #select_sql = r"SELECT tid, text  FROM tweets.tweet where tweet_lang <> 'en' order by tid desc limit 50;"
        #select_sql =  r"SELECT tid, text, url1  FROM tweets.tweet where tweet_lang = 'ar' order by tid desc   limit 1 ;"
        select_sql = r"SELECT tid, text, url1, tweet_lang  FROM tweets.tweet  order by tid desc   limit 1 ;"

        translate = Translator()

        while True:

            con = sql_twts.connect2database(database_paras)
            #results = sql_twts.select_db(r'SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED ;', con)
            results = sql_twts.select_db(select_sql, con)
            con.close()
            #print('results in transTwts: ', results)

            if len(results) < 1:
                time.sleep(10)
                continue
            df = pd.DataFrame(list(results), columns=['tweetID', 'text', 'url1', 'tweet_lang'])
            df = df.sort_values(by=['tweetID'], ascending=False)
            df['tweetID'] = df['tweetID'].astype(str)
            i = int(random.random() * len(df))
            #print('df: ', df.ix[0, 'url1'])
            #print('i: ', i)
            #df = df.iloc[i]
            texts = list(df['text'])
            #urls = list(df['url1'])

            #texts = texts[i:i+1] # randomly get 1 tweet
            #print('texts: ', texts)
            for j in range(len(texts)):
                texts[j] = re.sub(r'https{0,1}:\/\/t.co\/[a-zA-Z0-9]+', '', texts[j])
                texts[j] = re.sub(r'#', '', texts[j])
                texts[j] = re.sub(r'@[a-zA-Z0-9_]+', '', texts[j])
                #RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
                #texts[j] = RE_EMOJI.sub(r'', texts[j])
                texts[j] = emoji.demojize(texts[j])

            #print('texts: ', texts)
            trans = translate.translate(texts, dest)
            traneEn = translate.translate(texts, 'en')
            translateds = [i.text for i in trans]
            translateds_En = [i.text for i in traneEn]

            #url_list = url1.split(";")

            for j in range(len(texts)):
                #print(r'df[text]: ', df.ix[i, 'text'])   # restore the random tweet
                lang = df.ix[j, 'tweet_lang'].strip()
                try:
                    lang_full = LANGUAGES[lang].capitalize()
                except:
                    lang_full = 'Unknown'
                print(r'Tweets translation (original language is {}): {} {}'.format(lang_full, df.ix[j, 'text'], df.ix[j, 'url1'].replace(';', '  ')))
                #print(r'df[text]: ', df.ix[j, 'url1'].replace(';', '  '))
                print('Tweets translation (English): ', translateds_En[j])
                print('Tweets translation (Chinese): ', translateds[j])
                print('')

            # print(r'df[text]: ', df['text'])
            # print('translateds: ', translateds)

            time.sleep(20)
            #print('Translation(text): {} , {}'.format(trans.text, text))


            #
            # sql_cls = db_op.SQL_tweets()
            # images_ID = list(df['tweetID'])
            #
            # classified = list(df['classified'])
            # con = sql_twts.connect2database(database_paras)
            #
            # if len(labels) > 0:
            #     # tried_url = 3 : the tweet images have been classified.
            #     # print('Probs: ', type(probs))
            #     # print('Probs: ', probs)
            #     sql_cls.update_rows('tweet', ['tid', 'Flooded', 'classified', 'tried_url', 'Flooded_prob'],
            #                         [images_ID, flooded, classified, [3] * len(images_ID), probs], db_info_str)
            #     print('labels: ', labels)
            #
            #
            # con.close()
    #

    except Exception as e:
        print("Error in transTwts(): ", str(e))
        time.sleep(10)
        transTwts(configs, dest)

def Face_detection(configs):
    time.sleep(random.random()*15)
    #print('Face_detection() started. ')
    try:
        #NN_cls = ptFldNN.NeuralNetwork()
        global database_paras, con
        sql_twts = db_op.SQL_tweets()

        #NN_cls.setClassifier(configs)

        #sql_twts.insert_twts(tweets_tpl, con)

        host = configs['database']['host'].replace('"', '')
        user = configs['database']['user'].replace('"', '')
        password = configs['database']['password'].replace('"', '')
        db = configs['database']['db'].replace('"', '')

        db_info_list = []
        db_info_list.append(host)
        db_info_list.append(user)
        db_info_list.append(password)
        db_info_list.append(db)

        db_info_str = ','.join(db_info_list)

        database_paras = db_op.Database_parameters(host, user, password, db)

        select_sql = r"SELECT `time`, tid, userid, img_index, username  FROM tweets.yolo_results where faced = -1 and class_name = 'person' and area_pct>0.3 and img_index > 0 order by tid desc limit 50;"
        # tried_url = 2 : the tweets have downloaded images to be classified.
        path = configs['tweets_img_download']['image_folder'].replace('"', '')
        #print('path in face detection(): ', path)


        while True:
            start = time.time()
            try:
                #time.sleep(1)
                con = sql_twts.connect2database(database_paras)
                results = sql_twts.select_db(select_sql, con)
                con.close()

                if len(results) < 1:
                    time.sleep(10)
                    continue
                df = pd.DataFrame(list(results), columns=['time','tid', 'userid',  'img_index','username'])
                df = df.drop_duplicates(subset=['tid', 'img_index'])
                #print("df: ", df)
                df = df.sort_values(by=['tid'], ascending=False)
                df['tid'] = df['tid'].astype(str)
                df['paths'] = path
                df['dates'] = df['time'].astype(str).str[:10]
                df['dates'] = df['dates'].str.replace('-', '')

                df['paths'] = df.apply(joinPath, axis=1)
                df['paths'] = df['paths'] + '-' + df['img_index'].astype(str) + '.jpg'
                df['paths'] = df['paths'].str.replace('-1','')

                images = df['paths']
                #print('paths in face detection: ', df['paths'].str[-32:])
                # for i in range(len(results)):
                #     #print('return:', results[i][0])
                #     print('df: ', df['paths'][i])
                # #results = list(results)
                print('Face detector got : ', len(df), 'images of person from YOLO-v3 detection.', 'Starts at tweetID:', df.ix[0, 'tid'])
                print('')
                # print('type of return: ', type(df))
                # print('path: ', path)
                #print('returns: ', df)
                #time.sleep(5)
                sql_cls = db_op.SQL_tweets()
                images_ID = list(df['tid'])
                if len(df) > 0:
                    # faced = 1 : the tweet images have been face detected.
                    sql_cls.update_rows_no_primarykey_1c_1v('yolo_results', 'tid', images_ID, 'faced', 1, db_info_str)
                    #con.close()
                    #.sleep(random.random()*8)
                    #print('labels: ', labels)
                    #print('')



                face_results = fpp.getFacefrom_list(images)
                #print(face_results)
                df_faces = pd.read_csv(StringIO(face_results), dtype={'tid': str}, index_col=None,
                                        delimiter=',', error_bad_lines=False, low_memory=False)
                df_faces = df_faces[['tid','img_index','face_token','ethnicity','gender','age','glass','smile','e_sadness','e_neutral','e_disgust','e_anger','e_surprise','e_fear','e_happiness','b_female','b_male','m_close','m_mask','m_open','s_dark','s_stain','s_acne','s_health','h_yaw','h_pitch','h_roll','face_quality','r_width','r_top','r_left','r_height','area_pixel','blurness', 'portion']]
                df_faces = pd.merge(df_faces, df[['tid', 'time', 'userid', 'username']], on=['tid'])
                faces_tpl = list(df_faces.itertuples(index=False))
                # print('df_faces[tid]: ', df_faces['tid'])
                # print('df_faces[img_index]: ', df_faces['img_index'])
                # print('faces_tpl: ', faces_tpl)


                df['classified'] = 1
                classified = list(df['classified'])
                con = sql_twts.connect2database(database_paras)
                if len(face_results) > 0:
                                    # tried_url = 3 : the tweet images have been classified.
                    sql_cls.insert_fields('face', ['tid','img_index','face_token','ethnicity','gender','age','glass','smile','e_sadness','e_neutral','e_disgust','e_anger','e_surprise','e_fear','e_happiness','b_female','b_male','m_close','m_mask','m_open','s_dark','s_stain','s_acne','s_health','h_yaw','h_pitch','h_roll','face_quality','r_width','r_top','r_left','r_height','area_pixel','blurness','portion','time','userid','username'], faces_tpl, con)
                    #sql_cls.insert_fields('face', ['tid','img_index','face_token','ethnicity','gender','age','glass','smile','e_sadness','e_neutral','e_disgust','e_anger','e_surprise','e_fear','e_happiness','b_female','b_male','m_close','m_mask','m_open','s_dark','s_stain','s_acne','s_health','h_yaw','h_pitch','h_roll','face_quality','r_width','r_top','r_left','r_height','area_pixel'],
                    #                      faces_tpl, con)
                con.close()
            except Exception as e:
                print("Error in Face_detector() while loop: ", repr(e))

                time.sleep(1)
                continue
            end = time.time()
            print('Face detection (PID:{}) ran {:.2f} second, processed {} images ( {:.2f} images/second ), got {} faces in {} images.'.format(os.getpid(), end - start, len(df), len(df)/(end-start), len(df_faces), len(df_faces.groupby(['tid', 'img_index']))))
            print('')


    except Exception as e:
        print("Error in Face detection (): ", str(e))
        print('')
        time.sleep(5)
        Face_detection(configs)

def getFiles_from_tid_and_imgindex(first_images,img_cnts):
    all_images = []
    #i = 0
    for i in range(len(first_images)):

        all_images.append(first_images[i])
        for j in range(1, img_cnts[i]):
            name = first_images[i].replace('.jpg', '-' + str(j + 1) + '.jpg')
            all_images.append(name)
            #print('first_images[i], img_cnts[i]: ', first_images[i], img_cnts[i])

    return all_images

def Imgs_classify(configs):
    print('Imgs_classify() started. ')
    try:
        NN_cls = ptFldNN.NeuralNetwork()
        ptFldNN.setImageClassify(configs)
        global database_paras, con
        sql_twts = db_op.SQL_tweets()

        NN_cls.setClassifier(configs)

        #sql_twts.insert_twts(tweets_tpl, con)

        host = configs['database']['host'].replace('"', '')
        user = configs['database']['user'].replace('"', '')
        password = configs['database']['password'].replace('"', '')
        db = configs['database']['db'].replace('"', '')

        db_info_list = []
        db_info_list.append(host)
        db_info_list.append(user)
        db_info_list.append(password)
        db_info_list.append(db)

        db_info_str = ','.join(db_info_list)

        database_paras = db_op.Database_parameters(host, user, password, db)

        #ptFldNN.setClassifier(configs)


        trainedNN = NN_cls.GetModel_from_file(r'D:\Flooding\trained_model_49_vgg.pkl')
        #print('trainedNN: ', trainedNN)
        select_sql = r"SELECT `time`, tid, userid,img_cnt, username FROM tweets.tweet where tried_url = 2 order by tid desc limit 500;"
        # tried_url = 2 : the tweets have downloaded images to be classified.
        path = configs['tweets_img_download']['image_folder'].replace('"', '')
        #setImageClassify(configs)

        yolo = yolo_lib.YOLO()

        while True:
            try:
                time.sleep(5)
                con = sql_twts.connect2database(database_paras)
                results = sql_twts.select_db(select_sql, con)
                con.close()

                if len(results) < 1:
                    time.sleep(30)
                    continue
                df = pd.DataFrame(list(results), columns=['time', 'tid', 'userid', 'img_cnt', 'username'])
                df = df.sort_values(by=['tid'], ascending=False)
                df['tid'] = df['tid'].astype(str)
                df['paths'] = path
                #df['dates']
                df['dates'] = df['time'].astype(str).str[:10]
                df['dates'] = df['dates'].str.replace('-', '')
                df['paths'] = df.apply(joinPath, axis=1)
                df['paths'] = df['paths'] + '.jpg'

                # for i in range(len(results)):
                #     #print('return:', results[i][0])
                #     print('df: ', df['paths'][i])
                # #results = list(results)

                # print('type of return: ', type(df))
                # print('path: ', path)
                #print('returns: ', df[['tid', 'img_cnt']])
                #time.sleep(5)
                #images = list(df['paths'])
                images = getFiles_from_tid_and_imgindex(list(df['paths']), list(df['img_cnt']))
                #print('images: ', images)
                print('Classifier got : ', len(images), 'images.')
                # for jpg in images:     # check the image
                #     try:
                #         im = Image.open(jpg)
                #         im.verify()
                #     except Exception as err:
                #         print("Error in verify: ", err, jpg)
                        #print('deleting: ' + jpg)
                        #os.remove(jpg)

                labels, probs = NN_cls.path_list_infer(images, trainedNN)  # NN return 0 is flooded, then mysql set Flooded = 1.
                flooded = [1 - i for i in labels]

                #sql_cls = db_op.SQL_tweets()


                sql_cls = db_op.SQL_tweets()
                images_ID = list(df['tid'])
                df['classified'] = 1
                classified = list(df['classified'])
                con = sql_twts.connect2database(database_paras)

                if len(labels) > 0:
                    # tried_url = 3 : the tweet images have been classified.
                    #print('Probs: ', type(probs))
                    #print('Probs: ', probs)
                    sql_cls.update_rows('tweet', ['tid', 'Flooded', 'classified', 'tried_url', 'Flooded_prob'], [images_ID, flooded, classified, [3]*len(images_ID), probs], db_info_str)
                    print('labels: ', labels)
                    print('')

                # YOLO classifier

                yolo_results = yolo.detect_image_list(images, 1)
                if len(yolo_results) > 0:
                                    # tried_url = 3 : the tweet images have been classified.
                    #print('yolo_results: ', yolo_results)
                    df_yolo = yolo_results[['tid', 'top','left','bottom', 'right',
                                     'class_name', 'score',
                                    'out_class', 'area_pct', 'img_index']]
                    df_yolo['tid'] = df_yolo['tid'].astype(str)

                    # if len(df_yolo) > 0:
                    #     test = df_yolo.loc[df_yolo.img_index > 2]
                    #     print('df[img_index]: before  ', test)

                    df_yolo = pd.merge(df_yolo, df[['userid', 'tid', 'time', 'username']], on=['tid'])

                    # if len(df_yolo) > 0:
                    #     test = df_yolo.loc[df_yolo.img_index > 2]
                    #     print('df[img_index]: after ', test)
                    #print('df_yolo in classify():', df_yolo[['userid','tid', 'time', 'img_index']])
                    #print('df_yolo in classify():', df_yolo['img_index'])
                    yolo_tpl = list(df_yolo.itertuples(index=False))
                    sql_cls.insert_fields('yolo_results', ['tid', 'top', 'left', 'bottom', 'right', 'class_name', 'score', 'out_class', 'area_pct', 'img_index', 'userid',  'time', 'username'],
                                          yolo_tpl, con)


                con.close()
            except Exception as e:
                print("Error in Imgs_classify() while loop: ", repr(e))
                yolo.close_session()
                time.sleep(10)
                continue


    except Exception as e:
        print("Error in Imgs_classify(): ", str(e))
        yolo.close_session()
        time.sleep(10)
        Imgs_classify(configs)


if __name__ == '__main__':

    try:
        # PROCNAME = "python.exe"
        #
        # for proc in psutil.process_iter():
        #     # check whether the process name
        #     if proc.name() == PROCNAME:
       #        proc.kill()

        cfg_cls = config()
        global configs
        shared_list = mp.Manager().list()
        configs = cfg_cls.get_configs()

        #print('configs: ', configs)
        #print('BBOX: ', twts_dl.BBOX)
        #tweets_list = mp.Manager().list()
        #tweets_list = []
        # p_Face0 = Process(target=Face_detection, args=(configs,))
        # p_Face0.start()
        #
        # p_Face1 = Process(target=Face_detection, args=(configs,))
        # p_Face1.start()
        #
        # p_Face2 = Process(target=Face_detection, args=(configs,))
        # p_Face2.start()
        #
        # p_Face3 = Process(target=Face_detection, args=(configs,))
        # p_Face3.start()
        #
        # p_Face4 = Process(target=Face_detection, args=(configs,))
        # p_Face4.start()
        #
        # p_Face5 = Process(target=Face_detection, args=(configs,))
        # p_Face5.start()
        #
        # p_Face6 = Process(target=Face_detection, args=(configs,))
        # p_Face6.start()

        p_getTwts = Process(target=twts_download, args=(configs, shared_list))
        p_getTwts.start()


        p_getImgs = Process(target=Imgs_download, args=(configs, shared_list))
        p_getImgs.start()
        # #
        # p_Img_classify = Process(target=Imgs_classify, args=(configs,))
        # p_Img_classify.start()
        #
        # p_Trans = Process(target=transTwts, args=(configs, 'zh-CN'))
        # p_Trans.start()
        #
        # p_Face3 = Process(target=Face_detection, args=(configs,))
        # p_Face3.start()

    except Exception as err:
        print("Error in __main__(): ", err)
        cfg_cls = config()
        configs = cfg_cls.get_configs()

        print('configs: ', configs)
        print('BBOX: ', twts_dl.BBOX)
        # p_getTwts = Process(target=twts_download, args=(configs, ))
        # p_getTwts.start()
        # p_getTwts.join()

        p_getImgs = Process(target=Imgs_download, args=(configs, ))
        p_getImgs.start()
        # #
        # p_Img_classify = Process(target=Imgs_classify, args=(configs,))
        # p_Img_classify.start()
        #
        # p_Trans = Process(target=transTwts, args=(configs, 'zh-CN'))
        # p_Trans.start()

        # p_Face = Process(target=Face_detection, args=(configs,))
        # p_Face.start()
        #
        # p_Face1 = Process(target=Face_detection, args=(configs,))
        # p_Face1.start()
        #
        # p_Face2 = Process(target=Face_detection, args=(configs,))
        # p_Face2.start()
        #
        # p_Face3 = Process(target=Face_detection, args=(configs,))
        # p_Face3.start()

        print("Start over.")
    #print('BBOX: ', twts_dl.BBOX)


