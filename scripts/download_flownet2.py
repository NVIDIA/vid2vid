import os
from download_gdrive import *
import torch

"""if torch.__version__ == '0.4.1':
	file_id = '1gKwE1Ad41TwtAzwDcN3dYa_S6DcVyiSl'
	file_name = 'flownet2_pytorch_041.zip'
else:
	file_id = '1F2h_6e8gyTqxnbmFFW72zsxx_JX0dKFo'	
	file_name = 'flownet2_pytorch_040.zip'"""

chpt_path = './models/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
"""destination = os.path.join(chpt_path, file_name)
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)"""
os.system('cd %s/flownet2_pytorch/; bash install.sh; cd ../../' % chpt_path)