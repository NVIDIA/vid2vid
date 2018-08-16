import os
from download_gdrive import *

file_id = '16RhqUAtD_Aw29ZDyhW6IDDCxge7MGGHj'
chpt_path = './models/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'flownet2_pytorch.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)
os.system('cd %s/flownet2_pytorch/; bash install.sh; cd ../' % chpt_path)