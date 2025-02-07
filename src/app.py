# module app

# system
import os
import argparse
import traceback

# flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# computer vision
from yolov5 import detect

# init app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'src/static/uploads/'
app.config['RESULT_FOLDER'] = 'src/static/results/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# ensure upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        # get file
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # save the uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # process the image using YOLOv5
            detect.run(weights='spi_demo_yolov5l.pt', source=upload_path, project=app.config['RESULT_FOLDER'], name='', exist_ok=True)

            # return the result page with both images
            return render_template('index.html', original_image=filename, processed_image=filename)

    return render_template('index.html')


@app.route('/clear', methods=['POST'])
def clear():
    # Clear uploaded and result files
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return redirect(url_for('index'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', help='ip address or localhost', default='localhost', type=str)
    parser.add_argument('--port', help='bind specified port', default=8081, type=int)
    args = parser.parse_args()
    try:
        app.run(debug=True, host=args.host, port=args.port)
    except:
        traceback.print_exc()
    
    