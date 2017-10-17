# Recognition and Classification System

## References
- https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example
- https://github.com/datitran/object_detector_app

## Language
- Python 3.5.2

## Prerequisites
1. TensorFlow 1.3
2. TensorFlow Object Detection API
3. OpenCV 3.2
4. Flask 0.12.2

## Install packages

```
# apt-get update
# apt-get install -y protobuf-compiler python-pil python-lxml python-pip python-dev git
# pip install Flask==0.12.2 WTForms==2.1 Flask_WTF==0.14.2 Werkzeug==0.12.2
```

## Install the Object Detection API library

```
# cd /opt
# git clone https://github.com/tensorflow/models
# cd models/research
# protoc object_detection/protos/*.proto --python_out=.
```

## Download the pretrained model binaries

```
# mkdir -p /opt/graph_def
# cd /tmp
# for model in \
    ssd_mobilenet_v1_coco_11_06_2017 \
    ssd_inception_v2_coco_11_06_2017 \
    rfcn_resnet101_coco_11_06_2017 \
    faster_rcnn_resnet101_coco_11_06_2017 \
    faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017
  do \
    curl -OL http://download.tensorflow.org/models/object_detection/$model.tar.gz
    tar -xzf $model.tar.gz $model/frozen_inference_graph.pb
    cp -a $model /opt/graph_def/
  done
# ln -sf /opt/graph_def/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb /opt/graph_def/frozen_inference_graph.pb
```

## Running the application

```
# git clone https://github.com/anoopjeerige/realtimeCV
# cd ~/realtimeCV/obj_detect_multi
# export FLASK_APP=detector.app
# flask run
```
You have to wait around 60secs for the application to finish loading
 the pretrained model graph. You'll see the message
 `Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)` when it's ready.

Now you can access the instance's IP address using a web browser.
 When you upload an image file with a `jpeg`, `jpg`, or `png` extension,
 the application shows the result of the object detection inference.
 The inference may take up to 10 seconds, depending on the image.

You can check the objects by clicking labels shown to the right of the image.

You can upload an video file (preferably small < 5MB, .mp4, 3gp),
 for the object detection inference. The Real-time detection take the web cam
 as the source for the object detection.

## How to use different models
There are five pretrained models that can be used by the application.
 They have diffrent characteristics in terms of accuracy and speed.
 You can change the model used by the application by setting
 the PATH_TO_CKPT to point the frozen weights of the required model.

You specify one of the following models.

- ssd_mobilenet_v1_coco_11_06_2017
- ssd_inception_v2_coco_11_06_2017
- rfcn_resnet101_coco_11_06_2017
- faster_rcnn_resnet101_coco_11_06_2017
- faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017
