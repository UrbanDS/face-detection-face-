from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float
import sqlite3

# Global Variables
SQLITE                  = 'sqlite'
# https://medium.com/@mahmudahsan/how-to-use-python-sqlite3-using-sqlalchemy-158f9c54eb32
# Table Names
IMAGELIST           = 'imagelist'
FACEATTRIBUTES     = 'faceAttributes'

class MyDatabase:
    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {
        SQLITE: 'sqlite:///{DB}'
    }

    # Main DB Connection Ref Obj
    db_engine = None
    def __init__(self, dbtype, username='', password='', dbname=''):
        dbtype = dbtype.lower()
        if dbtype in self.DB_ENGINE.keys():
            engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = create_engine(engine_url)
            print(self.db_engine)
        else:
            print("DBType is not found in DB_ENGINE")
        print("DBType is  found in DB_ENGINE YAYYYYYY.....")

    def create_db_tables(self):
        metadata = MetaData()
        imagelist = Table(IMAGELIST, metadata,
                      Column('id', Integer, primary_key=True),
                      Column('imagename', String),
                      Column('imagepath', String),
                      Column('faceCount', Integer)

                      )
        faceAttributes = Table(FACEATTRIBUTES, metadata,
                        Column('id', Integer, primary_key=True),
                        Column('image_id', None, ForeignKey('imagelist.id')),
                        Column('gender', String),
                        Column('age', Integer),
                        Column('smile_value', Float),
                        Column('smile_threshold', Float),
                        Column('emotion_anger', Float),
                        Column('emotion_disgust', Float),
                        Column('emotion_fear', Float),
                        Column('emotion_happiness', Float),
                        Column('emotion_neutral', Float),
                        Column('emotion_sadness', Float),
                        Column('emotion_surprise', Float),
                        Column('ethnicity', String),
                        Column('beauty_male_score', Float),
                        Column('beauty_female_score', Float),
                        Column('mouthstatus_surgical_mask_or_respirator', Float),
                        Column('mouthstatus_other_occlusion', Float),
                        Column('mouthstatus_close', Float),
                        Column('mouthstatus_open', Float),
                        Column('skinstatus_health', Float),
                        Column('skinstatus_stain', Float),
                        Column('skinstatus_dark_circle', Float),
                        Column('skinstatus_acne', Float),
                        Column('face_rectangle_top', Integer),
                        Column('face_rectangle_left', Integer),
                        Column('face_rectangle_width', Integer),
                        Column('face_rectangle_height', Integer)
                )
        try:
            metadata.create_all(self.db_engine)
            print("Tables created")
        except Exception as e:
            print("Error occurred during Table creation!")
            print(e)

    # Insert, Update, Delete
    def execute_query(self, query=''):
        if query == '': return
        # print(query)
        with self.db_engine.connect() as connection:
            try:
                connection.execute(query)
            except Exception as e:
                print(e)
    # def sample_insert(self,):
    #     # Insert Data
    #     query = "INSERT INTO {}(id, first_name, last_name) " \
    #             "VALUES (3, 'Terrence','Jordan');".format(USERS)
    #     self.execute_query(query)
        # self.print_all_data(USERS)

    def insertmany_sqlite3(self,table='',columns='',data=''):
        for values in data:
            # query = "INSERT INTO " + table + ' (' + columns + ') VALUES ' + row + ";"
            query = "INSERT INTO {} ({}) VALUES ({});".format(table,columns,values)
            # print(query)
            self.execute_query(query)

    def get_all_data(self, table='', query=''):
        query = query if query != '' else "SELECT * FROM '{}';".format(table)
        # print(query)
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception as e:
                print(e)
            else:
                data = []
                for row in result:
                    # print(row)  # print(row[0], row[1], row[2])
                    data.append(row)
                result.close()
                return data

    def count(self,table=''):
        query = "SELECT COUNT(*) FROM {};".format(table)
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception as e:
                print(e)
            else:
                data = result
                # for row in result:
                #     # print(row)  # print(row[0], row[1], row[2])
                #     data.append(row)
                result.close()
                return data