#!/bin/bash

echo "------------------------------------------------"
echo "---- 1. CUSTIMIZED: SCRIPTS: DETECT: setup: ----"
echo "------------------------------------------------"
CUSTOMIZED_DETECT_BASH=${CUSTOMIZED_DETECT_BASH:-./customized/run-detect.sh}
if [ -s ${CUSTOMIZED_DETECT_BASH} ]; then
    echo "Found customized run-detect.sh found, use it instead of this..."
    #cd ./customized
    #./run-detect.sh
    ${CUSTOMIZED_DETECT_BASH}
    exit 0
fi
echo "... NOT FOUND: No './customized/run-detect.sh' script found -- USE Demo script: './run-detect.sh' ..."

echo "-------------------------------------------"
echo "---- 2. INPUT: WEIGHTS: FOLDER: setup: ----"
echo "-------------------------------------------"
echo ">>>> DEMO: using ${WEIGHTS_URL} Weights (trained model) ...."
WEIGHTS_URL=${WEIGHTS_URL:-https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt}
WEIGHTS_DIRECTORY=./weights/
WEIGHTS=${WEIGHTS_DIRECTORY}$(basename $WEIGHTS_URL) # yolov5s.pt
if [ ! -s ${WEIGHTS} ]; then
    echo ">>>> DOWNLOAD: Yolo pre-trained Model weights-and-bias: ${WEIGHTS_URL}"
    wget -c -P ${WEIGHTS_DIRECTORY} ${WEIGHTS_URL}
else
    echo ">>>> FOUND: Yolo pre-trained Model weights-and-bias: ${WEIGHTS}"
fi
echo "INPUT: WEIGHTS: FOLDER: ${WEIGHTS}"

echo "--------------------------------------"
echo "---- 3. INPUT: CONFIDENCE: setup: ----"
echo "--------------------------------------"
CONFIDENCE=${CONFIDENCE:-0.50}
echo "INPUT: OUPUT: CONFIDENCE: ${CONFIDENCE}"

echo "------------------------------------------"
echo "---- 4. INPUT: IMAGES: FOLDER: setup: ----"
echo "------------------------------------------"
echo ">>>> INPUT: IMAGES: FOLDER: ${SOURCE_IMAGES}"
echo ".... INPUT: IMAGES: CHECK: Any files in it? ...."
MY_SOURCE_IMAGES=${SOURCE_IMAGES}
if [ -n "$(ls -A ${SOURCE_IMAGES} 2>/dev/null)" ]; then
   echo ".... INPUT: IMAGES: FOUND: ${SOURCE_IMAGES}: Not empty: OK to use."
else
    # no files in it
    echo ".... INPUT: IMAGES: NOT FOUND: ${SOURCE_IMAGES} ! Use the 1st alternative folder: ./images/ (if not empty)"
    if [ ${SOURCE_IMAGES} != "./image" ] && [ -n "$(ls -A ./images 2>/dev/null)" ]; then
        SOURCE_IMAGES=./images
        echo ">>>> INPUT: IMAGES: FOUND: ${SOURCE_IMAGES}: Not empty: OK to use."
    else
        echo ".... INPUT: IMAGES: NOT FOUND: ./images ! Use the 2nd alternative folder: ./data/images/ (if not empty)" 
        SOURCE_IMAGES=./data/images
        if [ -n "$(ls -A ./data/images 2>/dev/null)" ]; then
            echo ">>>> INPUT: IMAGES: FOUND: ${SOURCE_IMAGES}: Not empty: OK to use."
        else
            echo "**** ERROR: Can't find any images files in: ${MY_SOURCE_IMAGES}, './images', or './data/images' folders! ABORT!"
            exit 1
        fi
    fi
fi
echo "INPUT: IMAGES: FOLDER:: ${SOURCE_IMAGES}"

echo "----------------------------------------"
echo "---- 5. OUTPUT: RUN: FOLDER: setup: ----"
echo "----------------------------------------"
## ---- Output directories setup: ---- ##
# -- outputs in <MY_PROJECT>/<MY_NAME>
MY_PROJECT=${MY_PROJECT:-runs/detect}
MY_NAME=${MY_NAME:-exp}
echo "OUTPUT: RUN: FOLDER: ${MY_PROJECT}/${MY_NAME}"

echo "----------------------------------------"
echo "---- 6. DETECT: IMAGES: RUN: setup: ----"
echo "----------------------------------------"

set -x

# Performance: GPU about 10~100 times faster than CPU:
python detect.py --source ${SOURCE_IMAGES} --weights ${WEIGHTS} --conf-thres ${CONFIDENCE} --save-txt --save-conf
# CPU
#python detect.py --source ${SOURCE_IMAGES} --device cpu --weights ${WEIGHTS} --conf-thres ${CONFIDENCE} --save-txt --save-conf

# JSON - not works (to-do: modify detect.py to support JSON)
#python detect.py --source ${SOURCE_IMAGES} --weights ${WEIGHTS} --conf ${CONFIDENCE} --save-json

set +x
