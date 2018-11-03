import os
from download_gdrive import *

file_id = '10LvNw-2lrh-6sPGkWbQDfHspkqz5AKxb'
chpt_path = './checkpoints/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'models_face.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)