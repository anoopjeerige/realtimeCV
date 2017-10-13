import io
import base64
import sys
import tempfile
import cv2

MODEL_BASE = '/home/anoop/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import Response
from flask import url_for
from flask import session
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
from cv2 import imencode
app = Flask(__name__)




PATH_TO_CKPT = '/home/anoop/tensorflow/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/mscoco_label_map.pbtxt'

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

# Helper Functions
def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image

def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)

def encode_image(image):
  image_buffer = io.BytesIO()
  image.save(image_buffer, format='PNG')
  mime_str = 'data:image/png;base64,'
  imgstr = '{0!s}'.format(base64.b64encode(image_buffer.getvalue()))
  quote_index = imgstr.find("b'")
  end_quote_index = imgstr.find("'", quote_index+2)
  imgstr = imgstr[quote_index+2:end_quote_index]
  imgstr = mime_str + imgstr
  #imgstr = 'data:image/png;base64,{0!s}'.format(
      #base64.b64encode(image_buffer.getvalue()))
  return imgstr

# Image class
class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])

class VideoForm(Form):
    input_video = FileField()

# Obect Dection Class
class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections.astype(int)

# Detection function
def detect_objects(image_path):
  image = Image.open(image_path).convert('RGB')
  boxes, scores, classes, num_detections = client.detect(image)
  image.thumbnail((480, 480), Image.ANTIALIAS)

  new_images = {}
  for i in range(num_detections):
    if scores[i] < 0.7: continue
    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    draw_bounding_box_on_image(new_images[cls], boxes[i],
                               thickness=int(scores[i]*10)-4)

  result = {}
  result['original'] = encode_image(image.copy())

  for cls, new_image in new_images.items():
    category = client.category_index[cls]['name']
    result[category] = encode_image(new_image)

  return result



@app.route('/')
def main_display():
    photo_form = PhotoForm(request.form)
    video_form = VideoForm(request.form)
    #return render_template('main.html', photo_form=photo_form, result={})
    return render_template('main.html', photo_form=photo_form, video_form=video_form, result={})

@app.route('/imgproc', methods=['GET', 'POST'])
def imgproc():
  video_form = VideoForm(request.form)
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))
  if request.method == 'POST' and form.validate():
    with tempfile.NamedTemporaryFile() as temp:
      form.input_photo.data.save(temp)
      temp.flush()
      print(temp.name)
      result = detect_objects(temp.name)

    photo_form = PhotoForm(request.form)
    return render_template('main.html',
                           photo_form=photo_form, video_form=video_form, result=result)
  else:
    return redirect(url_for('main_display'))

@app.route('/vidproc', methods=['GET', 'POST'])
def vidproc():
    print("In vidproc")
    form = VideoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST':
        print("vid sub")
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            form.input_video.data.save(temp)
            temp.flush()
            session['vid'] = temp.name
        return render_template('video.html')


@app.route('/vidpros')
def vidpros():
    graph = client.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')


    vid_source = cv2.VideoCapture(session['vid'])
    print("vid src")
    def generate(image_tensor, boxes, scores, classes, num_detections):
        ret, frame = vid_source.read()
        # tensor code



        print(ret)
        while(ret):
            #image_np = client._load_image_into_numpy_array(frame)
            image_np_expanded = np.expand_dims(frame, axis=0)

            (boxes_t, scores_t, classes_t, num_detections_t) = client.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes_t),
            np.squeeze(classes_t).astype(np.int32),
            np.squeeze(scores_t),
            client.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

            #image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')

            payload = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
            ret, frame = vid_source.read()

    print("Before return")
    return Response(generate(image_tensor, boxes, scores, classes, num_detections), mimetype='multipart/x-mixed-replace; boundary=frame')


client = ObjectDetector()

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(app)
    app.run(host='0.0.0.0', port=80, debug=False)
