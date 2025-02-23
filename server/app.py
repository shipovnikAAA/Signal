from flask import Flask, request, jsonify
import os
from datetime import datetime, timezon
import sqlite3
import matplotlib.pyplot
from tensorflow.keras.models import load_model
import numpy as np
import librosa.display
import matplotlib
matplotlib.use('agg')
from tensorflow_addons.metrics import F1Score
from keras.preprocessing import image
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


app = Flask(__name__)
# print('/'.join(__file__.split('\\')[:-1]))
dirname = os.path.abspath('temp')
db_path = os.path.abspath('database')
data_path = os.path.join(dirname, db_path, 'data')
CNN_BatchNormalization = load_model(r'C:\My_projects\nuke\models\CNN_with_BatchNormalization(best2)', 
                                    custom_objects={'F1Score': F1Score} )

def send_message(message : str = 'Казахстан угрожает нам ядерной бомбордировкой', Subject : str = 'Казахстан скинул ядерку на школу'):
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login("shipovniktuklosaw@gmail.com", "")
        
        msg = MIMEText(message)
        msg['Subject'] = Subject
        msg['From'] = formataddr(("Отправитель", "shipovniktuklosaw@gmail.com"))
        msg['To'] = formataddr(("Получатель", "Globalodessey@gmail.com"))
        
        server.sendmail("shipovniktuklosaw@gmail.com", ["Globalodessey@gmail.com"], msg.as_string())

class Database():
    def __init__(self, db_path=r'database\esp_database.db'):
        self.db_path = db_path

    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ESP (
                    uuid TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    time_init TEXT NOT NULL
                );
            ''')
            conn.commit()
        
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ESP_Sound (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    path TEXT NOT NULL,
                    file_sound TEXT NOT NULL,
                    file_spectrograms TEXT DEFAULT NULL,
                    esp_time TEXT NOT NULL,
                    server_time TEXT NOT NULL,
                    endpoint TEXT DEFAULT NULL,
                    FOREIGN KEY (path) REFERENCES ESP(path) ON DELETE CASCADE,
                    FOREIGN KEY (uuid) REFERENCES ESP(uuid) ON DELETE CASCADE
                );
            ''')
            conn.commit()

    def insert_data(self, uuid : str, uuid_dir : os.path, music_file : os.path, time_esp : str):
        time_server = datetime.now(timezone.utc).strftime('%Y-%m-%d %H-%M-%S %Z')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO ESP_Sound (uuid, path, file_sound, esp_time, server_time) VALUES (?,?,?,?,?)", (uuid, uuid_dir, music_file, time_esp, time_server))
            conn.commit()
    
    def insert_endpoint(self, uuid : str, endpoint : os.path, music_file : str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE ESP_Sound SET endpoint = ? WHERE uuid = ? and file_sound = ?", (endpoint, uuid, music_file))
            conn.commit()
    
    def insert_file_spectrograms(self, uuid : str, file_spectrograms : str, music_file : str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE ESP_Sound SET file_spectrograms =? WHERE uuid =? and file_sound =?", (file_spectrograms, uuid, music_file))
            conn.commit()
    
    class initEsp():
        def __init__(self, outer_instance):
            self.outer = outer_instance
        def init_esp(self, uuid : str, location : str, uuid_dir : os.path):
            time_server = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
            with sqlite3.connect(self.outer.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ESP (uuid, location, path, time_init) VALUES (?,?,?,?)", (uuid, location, uuid_dir, time_server))
                conn.commit()
        def find_esp(self, uuid : str) -> bool:
            try:
                with sqlite3.connect(self.outer.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT EXISTS(SELECT 1 FROM ESP WHERE uuid = ?)", (uuid,))
                    result = cursor.fetchone()[0]
                    return bool(result)
            except sqlite3.Error as e:
                print(f"Произошла ошибка при проверке primary key: {e}")
                return False
    def config_esp(self, location : str, uuid : str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE ESP SET location = ?, WHERE uuid = ?", (location, uuid))
            conn.commit()
        
def process(uuid : str, input_file : str, output_spectrograms : os.path, sound_path) -> str:
    def create_spectrogram(audio_file, image_file):
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        y, sr = librosa.load(audio_file)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)

        fig.savefig(image_file)
        matplotlib.pyplot.close(fig)
    
    create_spectrogram(input_file, output_spectrograms)
    db = Database()
    db.insert_file_spectrograms(uuid, output_spectrograms, input_file)
    image_spectorgram = image.img_to_array(image.load_img(output_spectrograms , target_size=(224, 224, 3)))
    
    predict_CNN_without_BatchNormalization  = CNN_BatchNormalization.predict(np.array([image_spectorgram]) / 255)
    print(predict_CNN_without_BatchNormalization)
    
    labels  = ['nuke', 'fireworks', 'other']
    
    predict_CNN_without_BatchNormalization = labels[np.argmax(predict_CNN_without_BatchNormalization)]
    
    return predict_CNN_without_BatchNormalization

@app.route('/')
def hello_world():
    return 'Hello, World!'

# @app.route('/audio/<name>')
# def greet(name):
#     return f'Hello, {name}!'


@app.route('/api/audio/', methods=['GET', 'POST'])
def add_sound():
    if request.method == 'POST':
        if 'music_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['music_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            uuid = request.form.get('uuid')
            time = request.form.get('time')
        except AttributeError:
            return jsonify({'error': 'Bad request'}), 400
        
        uuid_dir = os.path.join(data_path, uuid)
        if not os.path.exists(uuid_dir):
            return jsonify({'error': 'not found esp'}), 404
        
        sound_path = os.path.join(uuid_dir, 'sounds/' f"{uuid}_{time}.wav")
        print(sound_path)
        output_spectrograms = os.path.join(uuid_dir, 'spectrograms', f"{uuid}_{time}.png")
        
        if not os.path.exists(os.path.join(uuid_dir, 'sounds')):
            os.makedirs(os.path.join(uuid_dir, 'sounds'))
        if not os.path.exists(os.path.join(uuid_dir, 'spectrograms')):
            os.makedirs(os.path.join(uuid_dir, 'spectrograms'))
        
        file.save(sound_path)
        # data = request.data
        
        
        
        db = Database()
        initializer = db.initEsp(db)
        if initializer.find_esp(uuid):
            db.insert_data(uuid = uuid, uuid_dir = uuid_dir, music_file = sound_path, time_esp = time)
            endpoint = process(uuid, sound_path, output_spectrograms, os.path.join(uuid_dir, 'sounds'))
            db.insert_endpoint(uuid = uuid, endpoint = endpoint, music_file = sound_path)
            if endpoint == 'nuke':
                return jsonify({'warning': 'endpoint nuked'}), 200
            elif endpoint == 'fireworks':
                return jsonify({'warning': 'endpoint fireworks'}), 200
            else:
                return jsonify('ok'), 200
        else:
            return jsonify({'error': 'not found esp'}), 404

@app.route('/api/init/', methods=['GET', 'POST'])
def init_esp():
    if request.method == 'POST':
        try:
            uuid = request.form.get('uuid')
            location = request.form.get('location')
            # music_dir = request.form.get('music_dir')
        except AttributeError:
            return jsonify({'error': 'Bad request'}), 400
        uuid_dir = os.path.join(data_path, uuid)
        if not os.path.exists(uuid_dir):
            os.mkdir(uuid_dir)
        
        sounds_dir = os.path.join(uuid_dir, 'sounds')
        if not os.path.exists(sounds_dir):
            os.mkdir(sounds_dir)
        
        spectrograms_dir = os.path.join(uuid_dir,'spectrograms')
        
        if not os.path.exists(spectrograms_dir):
            os.mkdir(spectrograms_dir)
        
        db = Database()
        db.create_table()
        initializer = db.initEsp(db)
        if not initializer.find_esp(uuid):
            initializer.init_esp(uuid = uuid, location = location, uuid_dir = uuid_dir)
            return jsonify('ok'), 200
        else:
            return jsonify({'warninuuid_dirg': 'already initialized esp'}), 200



# if __name__ == '__main__':
#     app.run(debug=True)
#     app.run(host='0.0.0.0', port=5000)

# flask run -h 192.168.0.133 --port 5000 --debug