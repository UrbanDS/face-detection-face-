import MySQLdb
import pandas as pd
#import Twts_img_download as twts_img_dl
import csv
import shutil
class Database_parameters():
    host = 'localhost'
    user = 'root'
    password = 'huanning'
    db = 'tweets'

    def __init__(self, host, user, password, db):
        self.host = host
        self.user = user
        self.password = password
        self.db = db

class SQL_tweets():

    def getDBfrm_db_info_str(self, db_info_str):
        try:
            db_info_list = list(db_info_str.split(','))
            db_paras = Database_parameters(db_info_list[0], db_info_list[1], db_info_list[2], db_info_list[3])
            #db_paras = Database_parameters(db_info_list[0], db_info_list[1], db_info_list[2], db_info_list[3])
            sql_twts = SQL_tweets()
            return sql_twts.connect2database(db_paras)
        except Exception as e:
            print("Error in getDBfrm_db_info_str: ", repr(e))

    def connect2database(self, db_paras):
        #print(str(db_paras.password))
        try:
            con = MySQLdb.connect(host=db_paras.host,
                                  user=db_paras.user,
                                  password=db_paras.password,
                                  autocommit=True,
                                  db=db_paras.db,
                                  local_infile=1)
            return con

        except Exception as e:
            print('Error when connect to database {}: {}'.format(db_paras.db, str(e)))
            #pass

    def select_db(self, sql, conn):
        cursor = conn.cursor()
        cursor.execute('SET NAMES utf8mb4')
        cursor.execute("SET CHARACTER SET utf8mb4")
        cursor.execute("SET character_set_connection=utf8mb4")
        cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ ;")

        cursor.execute(sql)
        results = cursor.fetchall()
        #cursor.execute("COMMIT;")
        cursor.close()
        return results

    def insert_twts(self, tweets_tpl, con):
        try:
            cursor = con.cursor()
            con.set_character_set('utf8mb4')
            cursor.execute('SET NAMES utf8mb4;')
            cursor.execute('SET CHARACTER SET utf8mb4;')
            cursor.execute('SET character_set_connection=utf8mb4;')

            sql = "insert into twts_list (tweetID,userID,username,date,message,geoType,longitude,latitude,place,placeBboxWest,placeBboxEast,placeBboxSouth,placeBboxNorth,source,userMentions,urls,hashtags,retweetCount,userLocation,followersCount,friendsCount,joinDay,favouritesCount,language,statusesCount,replyToStatusId,replyToUserId,userVerified,userDescription,userUrl,favoriteCount,listedCount,placeType,bboxType,placeId,country_code,country,tweet_lang,message_en,message_cn,sentiment,topic) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            #print(sql)
            cursor.executemany(sql, tweets_tpl)
            #cursor.execute('insert twts_list (tweetID) values (5555);')
            con.commit()
            cursor.close()

           # sql = "insert into twts_list (tid,time,text,userid,url1,url2,Flooded,Flooding_Proof,Location_Verified,Moved,Online_Depth,Online_Inspection,Online_Analyst_Initials,Field_Depth,Field_Inspection,Field_Analyst_Initials,URL_Indication,Online_Notes,Field_Notes,lat,lon) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            #cursor.executemany(sql, tweets_tpl)

            #insert_sql = 'insert into twts_list (userDescription) values("\\xF0\\x9F\\x91\\xAB")'
            #cursor.execute(insert_sql)

        except Exception as e:
            print('Error in insert_twts: {}'.format(str(e)))
            #pass

    def insert_fields(self, table, fields, tweets_tpl, con): # more flexible than insert_twts().
        try:
            cursor = con.cursor()
            con.set_character_set('utf8mb4')
            cursor.execute('SET NAMES utf8mb4;')
            cursor.execute('SET CHARACTER SET utf8mb4;')
            cursor.execute('SET character_set_connection=utf8mb4;')

            fields_str = r'`' + r"`,`".join(fields) + r'`'
            s = r'%s,' * len(fields)#[:-1]
            s = s[:-1]
            # print("s: ", s)
            # print('(fields): ', (fields))
            # print('len(fields): ', len(fields))
            sql = "insert into {} ({}) values ({}) ".format(table, fields_str, s)
            #print('sql: ', str(sql))
            #print('tweets_tpl: ', str(tweets_tpl[0]))
            #tweetID,userID,username,date,message,geoType,longitude,latitude,place,placeBboxWest,placeBboxEast,placeBboxSouth,placeBboxNorth,source,userMentions,urls,hashtags,retweetCount,userLocation,followersCount,friendsCount,joinDay,favouritesCount,language,statusesCount,replyToStatusId,replyToUserId,userVerified,userDescription,userUrl,favoriteCount,listedCount,placeType,bboxType,placeId) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.executemany(sql, tweets_tpl)
            con.commit()
            cursor.close()

            #sql = "insert into twts_list (tid,time,text,userid,url1,url2,Flooded,Flooding_Proof,Location_Verified,Moved,Online_Depth,Online_Inspection,Online_Analyst_Initials,Field_Depth,Field_Inspection,Field_Analyst_Initials,URL_Indication,Online_Notes,Field_Notes,lat,lon) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            #cursor.executemany(sql, tweets_tpl)

            #insert_sql = 'insert into twts_list (userDescription) values("\\xF0\\x9F\\x91\\xAB")'
            #cursor.execute(insert_sql)

        except Exception as e:
            print('Error in insert_fields: {}'.format(str(e)))
            #pass

    def update_many(self, table=None, field_names=None, Key_values=None,  column_values=None, db_info_str=None):  # the first field is where clause id. Only one column can be updated.

        #print('arguments: ', locals().keys())
        #print('started to update: ')
        try:
            db_info_list = list(db_info_str.split(','))
            #print('db_info_list  in image update_rows(): ', db_info_list[0])
            #print(Key_values)
            if len(Key_values) > 1:
                tuples = list(zip(Key_values))
                #tuples = str(tuples)[1:-1]
            #print('tuples: ', tuples)
            #fields = ' = %s, '.join(field_names[1:]) + r' = %s'
            #print('fields: ', fields)

            names = field_names[1:]
            #print('name: ', names)
            sql_end = ','.join(names)  # A bug: a filed name cannot contain another field name, like Flooded and Flooded_prob.
            # for field in names:
            #     sql_end = sql_end.replace(field, field + "=VALUES(" + field + ")", 1)
            # print('sql_end: ', sql_end)
            update_sql = ("UPDATE {} SET {} = {}   WHERE {} = %s".format(table, field_names[1], column_values,  field_names[0]))
            #print('update_sql: ', update_sql, '\n')

            #update_sql = "UPDATE {} SET ".format(table, fields, tuples, sql_end)

            db_paras = Database_parameters(db_info_list[0], db_info_list[1], db_info_list[2], db_info_list[3])
            sql_twts = SQL_tweets()

            con = sql_twts.connect2database(db_paras)

            cursor = con.cursor()
            cursor.executemany(update_sql, tuples)
            con.commit()
            cursor.close()
            con.close()
        except Exception as e:
            print('Error in update_many(): ', str(e))

    def update_rows(self, table=None, field_names=None, column_values=None, db_info_str=None):

        #print('arguments: ', locals().keys())
        #print('started to update: ')
        try:
            db_info_list = list(db_info_str.split(','))
            #print('db_info_list  in image update_rows(): ', db_info_list[0])

            if len(column_values) > 1:
                tuples = list(zip(*column_values))
                tuples = str(tuples)[1:-1]
            #    print('tuples: ', tuples)
            fields = ','.join(field_names)
            #print('fields: ', fields)

            names = field_names[1:]
            #print('name: ', names)
            sql_end = ','.join(names)  # A bug: a filed name cannot contain another field name, like Flooded and Flooded_prob.
            for field in names:
                sql_end = sql_end.replace(field, field + "=VALUES(" + field + ")", 1)
            #print('sql_end: ', sql_end)
            update_sql = ("INSERT INTO {} "
                          " ({}) VALUES {}" +
                          " ON DUPLICATE KEY UPDATE {}"
                          ).format(table, fields, tuples, sql_end)
            #print('update_sql: ', update_sql + '\n')

            #update_sql = "UPDATE {} SET ".format(table, fields, tuples, sql_end)

            db_paras = Database_parameters(db_info_list[0], db_info_list[1], db_info_list[2], db_info_list[3])
            sql_twts = SQL_tweets()

            con = sql_twts.connect2database(db_paras)

            cursor = con.cursor()
            cursor.execute(update_sql)
            con.commit()
            cursor.close()
            con.close()
        except Exception as e:
            print('Error in update_rows(): ', str(e))

    def update_rows_no_primarykey_1c_1v(self, table=None, on_column=None, on_column_values=None, field_name=None, new_value=None, db_info_str=None):
        try:
            db_info_list = list(db_info_str.split(','))
            #print('db_info_list  in image update_rows(): ', db_info_list[0])
            column_values = ''
            if len(on_column_values) > 0:
                #tuples = list(*on_column_values)
                column_values = ','.join(on_column_values)
                #tuples = str(tuples)[1:-1]
                #print('tuples: ', column_values)


            # new_value = str, will cause bug
            #sql_end = ','.join(names)  # A bug: a filed name cannot contain another field name, like Flooded and Flooded_prob.
            select_sql = ("select * from `{}` " +
                          " WHERE {} in ({}) for update"
                          ";").format(table, on_column, column_values)
            #print('sql_end: ', sql_end)
            update_sql = ("UPDATE `{}` "
                          " SET `{}` = {}" +
                          " WHERE {} in ({})"
                          ";").format(table, field_name, new_value, on_column, column_values)
            #print('update_sql in update_rows_no_primarykey_1c_1v(): ', update_sql + '\n')

            db_paras = Database_parameters(db_info_list[0], db_info_list[1], db_info_list[2], db_info_list[3])
            sql_twts = SQL_tweets()

            con = sql_twts.connect2database(db_paras)

            cursor = con.cursor()
            cursor.execute('begin;')
            cursor.execute(select_sql)
            cursor.execute(update_sql)
            con.commit()
            cursor.close()
            con.close()
        except Exception as e:
            print('Error in update_rows_no_primarykey_1c_1v(): ', str(e))
if __name__ == '__main__':
    csv_path_name = r'E:\Tweets\test.csv'
    test_df = pd.read_csv(csv_path_name, quoting=csv.QUOTE_NONE, dtype={'tweetID': str}, delimiter=',', error_bad_lines=False,
                low_memory=False, encoding='utf8')
    test_df = twts_img_dl.cleanTwts(test_df)
    #test_df[pd.isnull(test_df)] = None
    tweets_tpl = list(test_df.itertuples(index=False))
    print('len of tweets_tpl: ', len(tweets_tpl))

    db_paras = Database_parameters('localhost', 'root', 'huanning', 'tweets')
    sql_twts = SQL_tweets()

    con = sql_twts.connect2database(db_paras)
    sql_twts.insert_twts(tweets_tpl, con)
    cursor = con.cursor()
    select_sql = "select * from twts_list"
    #cursor.executemany(select_sql, tweets_tpl)





    cursor.execute(select_sql)
    results = cursor.fetchall()
    i = 0
    con.commit()
    cursor.close()
    # for row in results:
    #     i += 1
    #     print(i, row)

    con.close()