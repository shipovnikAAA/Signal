import numpy as np
import librosa.display
import os
import matplotlib.pyplot as plt
import argparse

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_mp3(input_path, output_path, i, i_1):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir[i:i_1:]):
        input_file = input_path+'/'+file
        output_file = output_path+'/'+file.replace('.mp3', '.png')
        print(input_file, output_file)
        create_spectrogram(input_file, output_file)
        
# create_pngs_from_mp3('Sounds/concatenate', 'Spectrograms/concatenate')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('theme', type=str)
    parser.add_argument('i', type=int)
    parser.add_argument('i_1', type=int)
    args = parser.parse_args()

    create_pngs_from_mp3(f'Sounds/{args.theme}', f'Spectrograms/{args.theme}', args.i, args.i_1)