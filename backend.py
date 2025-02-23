import requests
from bs4 import BeautifulSoup
import os
import asyncio
import aiohttp
from threading import Thread
import time
import mutagen.mp3
import eyed3
from collections import defaultdict
import shutil
from pydub import AudioSegment
import pydub
import numpy as np
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import subprocess
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddGaussianSNR, Aliasing, AddShortNoises, BitCrush, BandPassFilter, GainTransition, RoomSimulator
from audiomentations import AddBackgroundNoise, PolarityInversion
import soundfile as sf
import subprocess
import os

class __parse__():
    def __init__(self,
                catigories : dict[str : list[str], str : list[str], str : list[str]] = {'nuke':[''],'fireworks':[''], 'other':[''], 'etalons':[''], 'concatenate':['']}, 
                urls : dict[str : dict[str:list[str]], str : dict[str:list[str]], str : dict[str:list[str]]] = {'zvukogram':{'nuke':[''], 'fireworks':[''], 'other':[''], 'etalons':[''], 'concatenate':['']}, 'zvukipro':{'nuke':[''], 'fireworks':[''], 'other':[''], 'etalons':[''], 'concatenate':['']}},
                direct_sounds: os.path = os.path.join(os.getcwd(), 'Sounds'),
                direct_spectrograms : os.path = os.path.join(os.getcwd(), 'Spectrograms'),
                base : bool =False,
                compiler : str = r"C:\Anaconda3\python.exe"):
        
        self.compiler = compiler
        self.data = []
        self.direct_sounds = direct_sounds
        self.direct_spectrograms = direct_spectrograms
        self.catigories = catigories
        self.base = base
        self.urls = urls
    
    class __process_data__():
        
        class __specrogram__():
            def __init__(self, input_path, output_path):
                self.input_path = input_path
                self.output_path = output_path

            def create_spectrogram(self, audio_file, image_file):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

                y, sr = librosa.load(audio_file)
                ms = librosa.feature.melspectrogram(y, sr=sr)
                log_ms = librosa.power_to_db(ms, ref=np.max)
                librosa.display.specshow(log_ms, sr=sr)

                fig.savefig(image_file)
                plt.close(fig)

            def create_pngs_from_mp3(self):
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)

                dir = os.listdir(self.output_path)

                for i, file in enumerate(dir):
                    input_file = os.path.join(self.output_path, file)
                    output_file = os.path.join(self.output_path, file.replace('.mp3', '.png'))
                    print(input_file, output_file)
                    self.create_spectrogram(input_file, output_file)
        
        class __rename__():
            def __init__(self, direct, theme, rel : str = '.mp3'):
                self.direct_sounds = os.path.join(direct, theme)
                self.theme = theme
                self.rel = rel
            def __process_before__(self):
                i = 0
                for filename in os.listdir(self.direct_sounds):
                    source = os.path.join(self.direct_sounds, filename)
                    destination = os.path.join(self.direct_sounds, f'{i}{self.rel}')
                    os.rename(source, destination)
                    i += 1
            def __process_after__(self):
                i = 0
                for filename in os.listdir(self.direct_sounds):
                    source = os.path.join(self.direct_sounds, filename)
                    destination = os.path.join(self.direct_sounds, f'{self.theme}_{i}{self.rel}')
                    os.rename(source, destination)
                    i += 1

            def __th_rename__(self):
                self.__process_before__()
                self.__process_after__()
        
        class __normalize__():
            def __init__(self, direct, theme):
                self.direct_sounds = os.path.join(direct, theme)
            def check_and_delete_corrupted_mp3(self):
            
                for filename in os.listdir(self.direct_sounds):
                    if filename.endswith(".mp3"):
                        filepath = os.path.join(self.direct_sounds, filename)
                        try:
                            mutagen.mp3.MP3(filepath)
                            # print(f"Файл '{filename}' в порядке.")
                        except Exception as e:
                            # print(f"Файл '{filename}' поврежден: {e}")
                            try:
                                os.remove(filepath)
                                print(f"Файл '{filename}' удален.")
                            except OSError as e:
                                print(f"Ошибка при удалении файла '{filename}': {e}")
            def __formated__(self, directoion_ffmpeg):
                
                directory = self.direct_sounds
                
                for filename in os.listdir(directory):
                    if filename.endswith(".mp3"):
                        filepath = os.path.join(directory, filename)
                        output_filepath = os.path.join(directory, "converted_" + filename)
                        
                        command = [
                            directoion_ffmpeg, 
                            '-i', filepath, 
                            '-vn', 
                            '-ar', '44100', 
                            '-ac', '2', 
                            '-ab', '192k', 
                            '-f', 'mp3',
                            output_filepath
                        ]
                        
                        try:
                            subprocess.run(command, check=True, capture_output=True, text=True)
                            print(f"File '{filename}' converted successfully.")
                        except subprocess.CalledProcessError as e:
                            print(f"Error converting '{filename}':")
                            print(f"Return code: {e.returncode}")
                            print(f"Stdout: {e.stdout}")
                            print(f"Stderr: {e.stderr}")
        
        class __duplicates__():
            def __init__(self, direct : os.path):
                self.direct_sounds = direct
                
            def find_and_remove_duplicates(self):
                """
                Находит и удаляет дубликаты MP3 файлов в указанной папке на основе названия композиции из метаданных.

                Args:
                    folder_path: Путь к папке с MP3 файлами.
                """

                track_titles = defaultdict(list)
                for filename in os.listdir(self.direct_sounds):
                    if filename.endswith(".mp3"):
                        filepath = os.path.join(self.direct_sounds, filename)
                        try:
                            audiofile = eyed3.load(filepath)
                            if audiofile.tag is not None and audiofile.tag.title is not None:
                                title = audiofile.tag.title.lower()
                                track_titles[title].append(filepath)
                                print(title)
                        except Exception as e:
                            print(f"Ошибка при чтении метаданных файла {filename}: {e}")


                for title, files in track_titles.items():
                    if len(files) > 1:
                        print(f"Найдены дубликаты для '{title}':")
                        for i, file in enumerate(files):
                            print(f"  {i+1}. {file}")

                        # оставляем первый файл, остальные удаляем
                        for file in files[1:]:
                            try:
                                os.remove(file)
                                print(f"  Файл '{file}' удален.")
                            except OSError as e:
                                print(f"  Ошибка при удалении файла '{file}': {e}")
            

        class __merge_mp3_files__():
            def __init__(self, base_mp3_path : os.path, mp3s : list[str], folder_path : os.path, output_folder_path : os.path, n : int = 0):
                self.base_mp3_path = base_mp3_path
                self.mp3s = mp3s
                self.n = n
                self.folder_path = folder_path
                self.output_folder_path = output_folder_path
            def __process__(self, folder_path, base_audios, mp3_file, output_folder_path, i):
                try:
                    audio_segment = AudioSegment.from_mp3(os.path.join(folder_path, mp3_file))
                    combined_audio = audio_segment
                    
                    change_first = int((len(audio_segment)/100)*15)
                    change_second = round((len(audio_segment)/100)*50)
                    num_overlays = random.randint(change_first, change_second)
                    max_position = len(audio_segment)
                    
                    while num_overlays >= 0:
                        choice = random.choice(self.mp3s)
                        # print((os.path.join(base_audios, choice)))
                        base_audio = AudioSegment.from_mp3(os.path.join(base_audios, choice))
                        max_position -= len(base_audio)
                        print(max_position)
                        
                        if max_position>0:
                            position = random.randint(0, max_position)
                            gain_change = random.uniform(-12, 35)
                            random_base_audio = base_audio + gain_change
                            combined_audio = combined_audio.overlay(random_base_audio, position=position)
                        num_overlays -= len(base_audio)
                    
                    output_file_path = os.path.join(output_folder_path, f"combined_{i}.mp3")
                    combined_audio.export(output_file_path, format="mp3")
                    print(f"Создан файл: {output_file_path}")
                except Exception as e:
                    print(f"Ошибка при обработке файла {mp3_file}: {e}")
            def merge_mp3_files(self):
                """
                Объединяет базовый MP3 файл с заданным количеством MP3 файлов из папки.
                Args:
                    base_mp3_path: Путь к базовому MP3 файлу.
                    folder_path: Путь к папке с MP3 файлами.
                    output_folder_path: Путь к папке для сохранения результатов.
                    num_files: Количество файлов для объединения (по умолчанию 100).
                """
                if not os.path.exists(self.output_folder_path):
                    os.makedirs(self.output_folder_path)
                mp3_files = [f for f in os.listdir(self.folder_path) if f.endswith('.mp3')]
                mp3_files.sort()
                th = []
                for i, mp3_file in enumerate(mp3_files):
                    th.append(Thread(target = self.__process__, args=(self.folder_path, self.base_mp3_path, mp3_file, self.output_folder_path, i+self.n)))
                for t in th:
                    t.start()
                    time.sleep(0.1)
                for t in th:
                    t.join()
    
    
        class __augment__():
            def __init__(self, pt):
                self.pt = pt
            def __start__(self):
                th = []
                for j in os.listdir(os.path.join(r'C:\My_projects\nuke\Sounds/', self.pt)):
                    waveform, sample_rate = librosa.load(os.path.join(r"C:\My_projects\nuke\Sounds/", self.pt, j))
                    for i in range(9):
                        t = Thread(target=self.process, args=(waveform, sample_rate, j, i, self.pt))
                        th.append(t)
                        t.start()
                        time.sleep(0.05)
                for t in th:
                    t.join()
            def process(waveform, sample_rate, j, i, pt):
                augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.8),
                # TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
                # PitchShift(min_semitones=-5, max_semitones=5, p=0.8),
                # Shift(p=1),
                AddGaussianSNR(
                    min_snr_db=5.0,
                    max_snr_db=10.0,
                    p=0.8
                ),
                Aliasing(min_sample_rate=8000, max_sample_rate=30000, p=0.8),

                AddShortNoises(
                    sounds_path=[r"C:\My_projects\nuke\Sounds\other", r"C:\My_projects\nuke\Sounds\fireworks", r"C:\My_projects\nuke\Sounds\nuke"],
                    min_snr_db=3.0,
                    max_snr_db=15.0,
                    noise_rms="relative_to_whole_input",
                    min_time_between_sounds=2.0,
                    max_time_between_sounds=8.0,
                    noise_transform=PolarityInversion(),
                    p=0.8
                ),
                AddBackgroundNoise(
                    sounds_path=[r"C:\My_projects\nuke\Sounds\other", r"C:\My_projects\nuke\Sounds\fireworks", r"C:\My_projects\nuke\Sounds\nuke"],
                    noise_rms="absolute",
                    min_absolute_rms_db=-45.0,
                    max_absolute_rms_db=-15.0,
                    noise_transform=PolarityInversion(),
                    p=0.8
                ),
                BitCrush(min_bit_depth=5, max_bit_depth=14, p=0.8),
                BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=0.8),
                GainTransition(
                min_gain_db = -10,
                max_gain_db = 10,
                min_duration = 0.7,
                max_duration  = 6,
                duration_unit = "seconds",
                p = 0.8
                ),
                RoomSimulator(use_ray_tracing=True, p=0.8),
                ])
                augmented_samples = augment(samples=waveform, sample_rate=sample_rate*2)
                sf.write(rf"C:\My_projects\nuke\Sounds\{pt}_concatenate\{pt}_{j}_{i}.mp3", augmented_samples, sample_rate) 
                print(j)
    
    class __zvukogram__():
        def __init__(self, i='', k : list[str] = [''], add_url: list[str] = [''], real_direct :os.path = os.path.join(os.getcwd(), 'Sounds')):
            """AI is creating summary for __init__

            Args:
                i (str, optional): [description]. Defaults to '' for theme.
                k (str, optional): [description]. Defaults to '' for question.
                real_direct (os.path, optional): [description]. Defaults to os.path.join(os.getcwd(), 'Sounds') for directory.
                add_url (list[str], optional): [description]. Defaults to [''] for additional downloading.
            """
            
            self.theme = i
            self.question = k
            self.url = f'https://zvukogram.com/?r=search&s='
            self.add_url = add_url
            self.direct_sounds = os.path.join(real_direct, self.theme)
            self.real_direct = real_direct
            print(self.direct_sounds)
        
        
        def __get_parse__(self):
            urls = []
            if self.question[0] or self.add_url[0]:
                if self.question[0]:
                    for q in self.question:
                        self.url += q
                        response = requests.get(self.url)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        for link in soup.find_all('span', class_ = 'dwdButtn'):
                            if link['data-type'] == 'mp3':
                                urls.append(link['data-id'])
                        # print(soup)
                        # for link in soup.find_all('a', href=True):
                        #     if link['href'].startswith('/index.php?r=site/download'):
                        #         if not link['href'].endswith('wav') and not link['href'].endswith('ogg') and not link['href'].endswith('m4r'):
                        #             urls.append(link['href'].split('&')[-1])
                
                if self.add_url[0]:
                    for self.url in self.add_url:
                        response = requests.get(self.url)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        for link in soup.find_all('span', class_ = 'dwdButtn'):
                            if link['data-type'] == 'mp3':
                                urls.append(link['data-id'])
                        # print(soup)
                        # for link in soup.find_all('a', href=True):
                        #     if link['href'].startswith('/index.php?r=site/download'):
                        #         if not link['href'].endswith('wav') and not link['href'].endswith('ogg') and not link['href'].endswith('m4r'):
                        #             urls.append(link['href'].split('&')[-1])
                print(urls)
                
                asyncio.run(self.main_pars(urls))
            # time.sleep(0.5)
            # __parse__.__rename__(self.direct_sounds, self.theme).__process_before__()
            # __parse__.__rename__(self.direct_sounds, self.theme).__process_after__()
        
        async def fetch_data(self, session, link:str):
            url = 'https://zvukogram.com/index.php?r=site/download&'+'id='+link
            async with session.get(url, ssl = False) as response:
                print(self.direct_sounds)
                with open(os.path.join(self.direct_sounds, f'{link}.mp3'), 'wb') as f:
                    while True:
                        chunk = await response.content.readany()
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Downloaded {link} from {url}")

            
        async def main_pars(self, urls:list):
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_data(session, link) for link in urls]
                results = await asyncio.gather(*tasks)
            return results
        
        def __base__(self):
            for theme, url in {'nuke':'https://zvukogram.com/category/zvuk-vyistrela/', 'fireworks':'https://zvukogram.com/category/zvuk-feyverka/', 'other':'https://zvukogram.com/category/zvuki-gorodskogo-shuma/'}.items():
                self.theme = theme
                self.direct_sounds = os.path.join(self.real_direct, self.theme)
                self.url = url
                response = requests.get(self.url)
                soup = BeautifulSoup(response.text, 'html.parser')
                urls = []
                
                for link in soup.find_all('a', href=True):
                    if link['href'].startswith('/index.php?r=site/download'):
                        if not link['href'].endswith('wav') and not link['href'].endswith('ogg') and not link['href'].endswith('m4r'):
                            urls.append(link['href'].split('&')[-1])
                
                asyncio.run(self.main_pars(urls))
                # time.sleep(0.5)
                # __parse__.__rename__(self.direct_sounds, self.theme).__process_after__()

    class __zvukipro__():
        def __init__(self, i='', add_urls=[''], real_direct=os.path.join(os.getcwd(), 'Sounds')):
            self.theme = i
            self.direct_sounds = os.path.join(real_direct, self.theme)
            self.real_direct = real_direct
            self.url = add_urls
            print(self.direct_sounds)

        def __get_parse__(self):
            if self.url[0]:
                for url in self.url:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    urls = []
                    for link in soup.find_all('a', href=True):
                        if link['href'].startswith('https://zvukipro.com/index.php?do=download&'):
                            urls.append(link['href'].split('&')[-1])
                    asyncio.run(self.main_pars(urls))
            # time.sleep(0.5)
            # __parse__.__rename__(self.direct_sounds).__process_before__()
            # __parse__.__rename__(self.direct_sounds, self.theme).__process_after__()
        
        async def fetch_data(self, session, link:str):
            url = 'https://zvukipro.com/index.php?do=download&'+link
            async with session.get(url, ssl = False) as response:
                print(self.direct_sounds)
                with open(os.path.join(self.direct_sounds, f'{link}.mp3'), 'wb') as f:
                    while True:
                        chunk = await response.content.readany()
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Downloaded {link} from {url}")

            
        async def main_pars(self, urls:list):
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_data(session, link) for link in urls]
                results = await asyncio.gather(*tasks)
            return results

        def __base__(self):
            for theme, url in {'nuke':['https://zvukipro.com/oryjie/246-zvuki-vystrelov-iz-oruzhiya.html',
                            'https://zvukipro.com/oryjie/844-zvuki-pistoleta-makarova.html',
                            'https://zvukipro.com/oryjie/798-zvuki-pistoleta-pulemeta-tompsona.html',
                            'https://zvukipro.com/oryjie/817-zvuki-desert-eagle-deagle.html',
                            'https://zvukipro.com/oryjie/833-zvuki-koktejlja-molotova.html'],
                        'fireworks': ['https://zvukipro.com/oryjie/247-zvuki-fireverk.html'], 
                        'other':['https://zvukipro.com/atmosphera/63-zvuki-goroda-i-gorodskogo-shuma.html',
                            'https://zvukipro.com/atmosphera/4160-atmosfernye-zvuki-kazahstana.html',
                            'https://zvukipro.com/atmosphera/4084-atmosfernye-zvuki-novoj-zelandii.html',
                            'https://zvukipro.com/transport/4249-zvuki-proezzhajuschej-mimo-mashiny-s-gudkom.html']}.items():
                self.theme = theme
                self.direct_sounds = os.path.join(self.real_direct, self.theme)
                urls = []
                for self.url in url:
                    response = requests.get(self.url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        if link['href'].startswith('https://zvukipro.com/index.php?do=download&'):
                            urls.append(link['href'].split('&')[-1])
                asyncio.run(self.main_pars(urls))
                
                # __parse__.__rename__(self.direct_sounds, self.theme).__process_after__()

    def __start_parse__(self):
        if self.base:
            th =[]
            th1 =[]
            self.data = self.__zvukogram__().__base__
            th.append(Thread(target=self.data, args=()))
            
            self.data1 = self.__zvukipro__().__base__
            th1.append(Thread(target=self.data, args=()))
            
            for t in th:
                t.start()
            for t in th1:
                t.start()
            
            for t in th:
                t.join()
            for t in th1:
                t.join()
        else:
            th1 = {'zvukigram':[], 'zvukipro':[]}
            for i, k in self.urls['zvukogram'].items():
                print(self.catigories[i], k, i)
                self.data = self.__zvukogram__(i, self.catigories[i], k).__get_parse__
                self.data = Thread(target=self.data, args=())
                th1['zvukigram'].append(self.data)
            
            for i, k in self.urls['zvukipro'].items():
                print(i,k)
                self.data = self.__zvukipro__(i, k).__get_parse__
                self.data = Thread(target=self.data, args=())
                th1['zvukipro'].append(self.data)
            
            for _, h in th1.items():
                for t in h:
                    t.start()
            
            for _, h in th1.items():
                for t in h:
                    t.join()
        
        time.sleep(1)
        self.__name_f__()
        
        th = []
        for i, k in self.catigories.items():
            self.data = self.__process_data__.__cheacker__(self.direct_sounds, i).check_and_delete_corrupted_mp3
            th.append(Thread(target=self.data, args=()))
        for t in th:
            t.start()
        for t in th:
            t.join()


        # self.data2.join()
        # self.data = self.__zvukogram__('nuke', 'Выстрелы', self.direct_sounds, self.base).__get_parse__()
    def __corrupted_mp3__(self, theme : list[str] = ['all'], directoion_ffmpeg : str = r'ffmpeg\ffmpeg.exe'):
        th = []
        th1 = []
        self.__name_f__(direct='sounds')
        if 'all' in theme :
            for i in self.catigories:
                th.append(Thread(target = (self.__process_data__.__normalize__(os.path.join(self.direct_sounds), i).check_and_delete_corrupted_mp3), args=()))
                th1.append(Thread(target = (self.__process_data__.__normalize__(os.path.join(self.direct_sounds), i).__formated__), args=(directoion_ffmpeg, )))
        
        else:
            for i in theme:
                th.append(Thread(target = (self.__process_data__.__normalize__(os.path.join(self.direct_sounds), i).check_and_delete_corrupted_mp3), args=()))
                th1.append(Thread(target = (self.__process_data__.__normalize__(os.path.join(self.direct_sounds), i).__formated__), args=(directoion_ffmpeg, )))
        
        for t in th:
            t.start()
        for t in th:
            t.join()
        
        for t in th1:
            t.start()
        for t in th1:
            t.join()
        

    def __concatenate_mp3__(self, etalons : list[str] = ['all'], inputs: list[str] = ['all'], outputs: str = 'concatenate', etalons_path : str = 'etalons'):
        th = []
        n = 0
        self.__process_data__.__rename__(os.path.join(self.direct_sounds), outputs, '.mp3').__process_before__()
        temp_catigories = self.catigories.copy()
        etalons_data = os.path.join(self.direct_sounds, etalons_path)
        del temp_catigories['concatenate'], temp_catigories['etalons']
        if 'all' in etalons:
            if 'all' in inputs:
                # for i in os.listdir(os.path.join(self.direct_sounds, etalons_path)):
                # if 'all' in inputs:
                for intuput_f in temp_catigories:
                    th.append(Thread(target = (self.__process_data__.__merge_mp3_files__(etalons_data, os.listdir(os.path.join(self.direct_sounds, etalons_path)), os.path.join(self.direct_sounds, intuput_f), os.path.join(self.direct_sounds, outputs), n).merge_mp3_files), args=()))
                    n += 1000
            else:
                # for i in os.listdir(os.path.join(self.direct_sounds, etalons_path)):
                for intuput_f in inputs:
                    # if i.endswith('.inputs') and i.startswith('etalon'):
                    th.append(Thread(target = (self.__process_data__.__merge_mp3_files__(etalons_data, os.listdir(os.path.join(self.direct_sounds, etalons_path)), os.path.join(self.direct_sounds, intuput_f), os.path.join(self.direct_sounds, outputs), n).merge_mp3_files), args=()))
                    n += 1000
        else:
            if 'all' in inputs:
                # for etalon in etalons:
                # if 'all' in inputs:
                for intuput_f in temp_catigories:
                    th.append(Thread(target = (self.__process_data__.__merge_mp3_files__(etalons_data, etalons, os.path.join(self.direct_sounds, intuput_f), os.path.join(self.direct_sounds, outputs), n).merge_mp3_files), args=()))
                    n += 1000
            else:
                # for etalon in etalons:
                for intuput_f in inputs:
                        th.append(Thread(target = (self.__process_data__.__merge_mp3_files__(etalons_data, etalons, os.path.join(self.direct_sounds, intuput_f), os.path.join(self.direct_sounds, outputs), n).merge_mp3_files), args=()))
                        n += 1000
        for t in th:
            t.start()
        for t in th:
            t.join()
        del temp_catigories
        self.__process_data__.__rename__(os.path.join(self.direct_sounds), outputs, '.mp3').__process_after__()

    def __name_f__(self, theme : list[str] = ['all'], direct : str = 'all'):
        th = []
        if direct == 'all':
            if 'all' in theme :
                for i in self.catigories:
                    self.data = self.__process_data__.__rename__(self.direct_sounds, i, '.mp3').__th_rename__
                    th.append(Thread(target=self.data, args=()))
                    self.data = self.__process_data__.__rename__(self.direct_spectrograms, i, '.png').__th_rename__
                    th.append(Thread(target=self.data, args=()))
            else:
                for i in theme:
                    self.data = self.__process_data__.__rename__(self.direct_sounds, i, '.mp3').__th_rename__
                    th.append(Thread(target=self.data, args=()))
                    self.data = self.__process_data__.__rename__(self.direct_spectrograms, i, '.png').__th_rename__
                    th.append(Thread(target=self.data, args=()))
        elif direct.lower() == 'sounds':
            if 'all' in theme :
                for i in self.catigories:
                    self.data = self.__process_data__.__rename__(self.direct_sounds, i, '.mp3').__th_rename__
                    th.append(Thread(target=self.data, args=()))
            else:
                for i in theme:
                    self.data = self.__process_data__.__rename__(self.direct_sounds, i, '.mp3').__th_rename__
                    th.append(Thread(target=self.data, args=()))
        else:
            if 'all' in theme :
                for i in self.catigories:
                    self.data = self.__process_data__.__rename__(self.direct_spectrograms, i, '.png').__th_rename__
                    th.append(Thread(target=self.data, args=()))
            else:
                for i in theme:
                    self.data = self.__process_data__.__rename__(self.direct_spectrograms, i, '.png').__th_rename__
                    th.append(Thread(target=self.data, args=()))
        for t in th:
            t.start()
        for t in th:
            t.join()

    def __duplicates__(self, theme : list[str] = ['all']):
        th = []
        if 'all' in theme:
            for i in self.catigories:
                th.append(Thread(target = (self.__process_data__.__duplicates__(os.path.join(self.direct_sounds, i)).find_and_remove_duplicates), args=()))
        else:
            for i in theme:
                th.append(Thread(target = (self.__process_data__.__duplicates__(os.path.join(self.direct_sounds, i)).find_and_remove_duplicates), args=()))
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.__name_f__(theme=theme)
    
    def __specrogram__(self, theme : list[str] = ['all']):
        th = []
        self.__name_f__(theme=theme)
        if 'all' in theme :
            for i in self.catigories:
                th.append(Thread(target = (self.__process_data__.__specrogram__(os.path.join(self.direct_sounds, i), os.path.join(self.direct_spectrograms, i)).create_pngs_from_mp3), args=()))
        else:
            for i in theme:
                th.append(Thread(target = (self.__process_data__.__specrogram__(os.path.join(self.direct_sounds, i)).create_pngs_from_mp3), args=()))
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.__name_f__(theme=theme)
    
    def __spectrograms_main_thread__(self, theme : list[str] = ['all'], batch_size : int = 400, first : list = 0):
        python_script = "spectrograms.py"
        
        self.__name_f__(theme=theme)
        if 'all' in theme :
            for i_theme in self.catigories:
                for i_theme in os.listdir('Sounds'):
                    num_files = len(os.listdir(f'Sounds/{i_theme}'))
                    for i in range(num_files // batch_size):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        if end >= first:
                            print(start, end)
                            subprocess.Popen([self.compiler, python_script, str(i_theme), str(start), str(end)])
                    
                    remaining_files = num_files % batch_size
                    if remaining_files > 0 and num_files >= first:
                        start = num_files - remaining_files
                        end = num_files
                        print(start, end)
                        subprocess.Popen([self.compiler, python_script, str(i_theme), str(start), str(end)])

        else:
            for i_theme in theme:
                for i in range(len(os.listdir(f'Sounds/{i_theme}'))//batch_size):
                    if (i+1)*batch_size >= first:
                        print(i*batch_size, (i+1)*batch_size)
                        subprocess.Popen([self.compiler, python_script, str(i_theme), str(int(i*batch_size)), str(int((i+1)*batch_size))])
    def __satrt_augment__(self, theme):
        th = []
        self.__name_f__(theme=theme)
        if 'all' in theme :
            for i in self.catigories:
                th.append(Thread(target = (self.__process_data__.__augment__(os.path.join(self.direct_sounds, i))).__start__), args=())
        else:
            for i in theme:
                th.append(Thread(target = (self.__process_data__.__specrogram__(os.path.join(self.direct_sounds, i)).create_pngs_from_mp3), args=()))
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.__name_f__(theme=theme)
        # self.__process_data__.__augment__().__start__()
        
        
# __parse__(catigories = {'nuke':['Выстрелы очередью'],'fireworks':[''], 'other':['Афоня', 'сон', 'Вздохи', 'Английский алфавит', 'Женская речь', 'Кашель', 'плач', 'Автобус']}).__start_parse__()

# __parse__(catigories= { 'nuke': [''],'fireworks': [''],'other': ['ельцин', 'депутаты'],'etalons': [''],'concatenate': [''] }, urls= { 'zvukogram': { 'nuke': [''],'fireworks': [''],'other': ['https://zvukogram.com/category/zvuk-putina/'] },'zvukipro': { 'nuke': [''],'fireworks': [''],'other': [''] } }).__start_parse__()

# __parse__().__duplicates__()

# __parse__().__concatenate_mp3__()

# __parse__(compiler = r'C:\anaconda3\envs\protect_of_terrorist_attacks\python.exe').__spectrograms_main_thread__()

__parse__().__corrupted_mp3__()

# __parse__().__duplicates__()
