# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
#FROM nvcr.io/nvidia/pytorch:21.03-py3
FROM python:3.8

MAINTAINER DrSnowbird "DrSnowbird@openkbs.org"

USER 0
##############################################
#### ---- Installation Directories   ---- ####
##############################################
ENV INSTALL_DIR=${INSTALL_DIR:-/usr/src}
ENV SCRIPT_DIR=${SCRIPT_DIR:-$INSTALL_DIR/scripts}

##############################################
#### ---- Corporate Proxy Auto Setup ---- ####
##############################################
#### ---- Transfer setup ---- ####
RUN mkdir -p ${SCRIPT_DIR} && ls -al ${SCRIPT_DIR}
COPY ./scripts ${SCRIPT_DIR}
RUN ls -al ${SCRIPT_DIR} 
RUN chmod +x ${SCRIPT_DIR}/*.sh

#### ---- Apt Proxy & NPM Proxy & NPM Permission setup if detected: ---- ####
RUN cd ${SCRIPT_DIR} ; ${SCRIPT_DIR}/setup_system_proxy.sh

# Install linux packages
RUN apt-get update -y
RUN apt-get install -y vim zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN python3 -m pip install --no-cache -r requirements.txt
RUN python3 -m pip install --no-cache coremltools onnx gsutil notebook
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app
#RUN ls -al /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app

ENTRYPOINT ["./docker-entrypoint.sh"]
# Default run-detect.sh (It will detect the existence of ./customized/run-detect.sh, then run it instead)
CMD ["./run-detect.sh"]
# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/train/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
