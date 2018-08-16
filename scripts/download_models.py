import os
from download_gdrive import *

file_id = '1MKtImgtnGC28EPU7Nh9DfFpHW6okNVkl'
chpt_path = './checkpoints/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'models.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)