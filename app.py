import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cStringIO as StringIO
import urllib
import exifutil
import darknet
import cv2 as cv

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/objdet_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif', 'tif', 'tiff'])

# Obtain the flask app object
app = flask.Flask(__name__)


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def draw_predict(img, predict):
    open_cv_image = np.array(img)
    for idx, item in enumerate(predict[1]):
        print(idx)
        print(item)
        x, y, w, h = item[2]
        half_w = w / 2
        half_h = h / 2
        text = str(idx + 1)+": "+item[0]
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        cv.rectangle(open_cv_image, (int(x - half_w), int(y - half_h)), (int(x + half_w), int(y + half_h)), (0,255,0), 2)
        cv.rectangle(open_cv_image, (int(x - half_w), int(y - half_h)), (int(x - half_w+text_width+2), int(y - half_h+text_height+2)), (0,255,0), cv.FILLED)
        cv.putText(open_cv_image, text, (int(x - half_w), int(y - half_h + text_height)), font, font_scale, (255, 0, 0), 1, cv.LINE_AA)
    return Image.fromarray(open_cv_image, 'RGB')

def embed_image_html(image_pil):
    """Creates an image embedded in HTML base64 format."""
    width, height = image_pil.size
    if height > 700:
        height = 700
        width = width * 700 / image_pil.size[1]
    if width > 700:
        width = 700
        height = height * 700 / image_pil.size[0]
    size = (width, height)
    resized = image_pil.resize(size)
    string_buf = StringIO.StringIO()
    resized.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

# Detect image from url
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        # download
        raw_data = urllib.urlopen(imageurl).read()
        string_buffer = StringIO.StringIO(raw_data)
        image_pil = Image.open(string_buffer)
        filename = os.path.join(UPLOAD_FOLDER, 'tmp.jpg')
        with open(filename,'wb') as f:
            f.write(raw_data)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template('index.html', has_result=True, result=(False, 'Cannot open image from URL.'))

    logging.info('Image: %s', imageurl)

    results = app.clf.classify_image(filename)
    predict_img = draw_predict(image_pil, results)
    new_img_base64 = embed_image_html(image_pil)
    return flask.render_template('index.html', has_result=True, result=results, imagesrc=new_img_base64)


# Dectect image from upload
@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image_pil = exifutil.open_oriented_pil(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    results = app.clf.classify_image(filename)
    predict_img = draw_predict(image_pil, results)
    new_img_base64 = embed_image_html(predict_img)
    return flask.render_template('index.html', has_result = True, result = results, imagesrc = new_img_base64)


class ImagenetClassifier(object):
    default_args = {
         'model_def_file': (
             '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
         'pretrained_model_file': (
             '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
         'mean_file': (
             '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
         'class_labels_file': (
             '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
         'bet_file': (
             '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }

    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    # fyk
    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        self.net = darknet.load_net("cfg/MBABT.cfg", "MBABT_final.weights", 0)
        self.meta = darknet.load_meta("cfg/MBABT.data")


    def classify_image(self, image_filename):
        try:
            starttime = time.time()
            # scores = self.net.predict([image], oversample=True).flatten()
            results = darknet.detect(self.net, self.meta, image_filename)
            # [('F22', 0.9006772637367249, (338.6946105957031, 431.28515625, 608.9721069335938, 220.40663146972656)),
            #  ('F22', 0.890718400478363, (545.9476318359375, 294.4508361816406, 509.1690979003906, 177.72409057617188)),
            #  ('F22', 0.8847938179969788, (642.2884521484375, 193.6743927001953, 401.5226745605469, 135.20948791503906))]
            endtime = time.time()
            bet_result = [(str(idx+1)+' : '+v[0], '%.5f' % v[1])
                          for idx, v in enumerate(results)]
            # logging.info('bet result: %s', str(bet_result))
            rtn = (True, results, bet_result, '%.3f' % (endtime - starttime))
            return rtn

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    #app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
