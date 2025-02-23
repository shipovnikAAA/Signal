import requests
from datetime import datetime
import pytz
from threading import Thread
import time

adress = 'http://192.168.38.193:5000'
# adress = 'http://192.168.0.133:5000'
def send_audio(adress):
    #print('attempting to send audio')
    url = adress +'/api/audio/'
    with open(r'C:\My_projects\nuke\Sounds\samples\Chiikawa Usagi.mp3', 'rb') as file:
        tz = pytz.timezone('UTC') 
        time_esp = datetime.now(tz).strftime('%Y-%m-%d %H-%M-%S %Z')
        data = {'uuid':'-jx-1', 'time': time_esp}
        files = {'music_file': file}

        req = requests.post(url, files=files, data=data)
        print(req.status_code)
        print(req.text)
        
def send_audio1(adress):
    #print('attempting to send audio')
    url = adress +'/api/audio/'
    with open(r'C:\My_projects\nuke\Sounds\samples\IMG_9658.mp3', 'rb') as file:
        tz = pytz.timezone('UTC') 
        time_esp = datetime.now(tz).strftime('%Y-%m-%d %H-%M-%S %Z')
        data = {'uuid':'-jx-1', 'time': time_esp}
        files = {'music_file': file}

        req = requests.post(url, files=files, data=data)
        # print(req.status_code)
        print(req.json())

def init_esp(adress):
    url = adress + '/api/init/'
    data = {'uuid':'-jx-1', 'location': 'rusnya'}
    req = requests.post(url, data=data)
    print(req.status_code)
    print(req.text)

# Threads
th = []
for _ in range(10):
    th.append(Thread(target=send_audio, args=(adress,)))
    th.append(Thread(target=send_audio1, args=(adress,)))
for t in th:
    time.sleep(0.1)
    t.start()
for t in th:
    t.join()
# init_esp(adress)

