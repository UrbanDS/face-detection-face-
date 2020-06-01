from PIL import Image
import face_recognition
import glob
from database import mydatabase

dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='mydb.sqlite')
# dbms.create_db_tables()

def findFace(filenames):
# Load the jpg file into a numpy array

    for file in filenames:
        data = []
        all_data = []
        image = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(image)
        # print("I found {} face(s) in this photograph.".format(len(face_locations)))
        if(len(face_locations)>0):
            data = ["'"+file+"'","'"+file+"'",str(len(face_locations))]
            data_string = ",".join(data)
            all_data.append(data_string)
        # for face_location in face_locations:
        #     top, right, bottom, left = face_location
        #     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        dbms.insertmany_sqlite3("imagelist","imagename,imagepath,faceCount",all_data)
    print("--------- completed")
    # You can access the actual face itself like this:
    #     face_image = image[top:bottom, left:right]
        # pil_image = Image.fromarray(face_image)
        # pil_image.show()


# filenames = glob.glob('atlanta-beltline/photos/*.jpg')
filenames = glob.glob('nyc/photos/*.jpg')

findFace(filenames)

