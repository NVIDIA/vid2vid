#Thanks @dustinfreeman for providing the script

#Install docker-ce https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository
sudo apt-get install -y \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce


#Install nvidia-docker2 https://github.com/NVIDIA/nvidia-docker
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd


#NVIDIA drivers
#This triggers an interactive request to the user.
#Would love an alternative!
DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y keyboard-configuration
sudo apt install -y ubuntu-drivers-common

apt-get install -y nvidia-384

#Reboot so the nvidia driver finishes install
sudo reboot



