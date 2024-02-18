import os
import torchaudio
from torchaudio.transforms import Resample
from glob import glob
import random
import json

def random_sampling(waveform_list, label_list, resample):
    sampled_waveform_list = []
    sampled_label_list = []
    for wave,label in zip(waveform_list,label_list):

        files_wav = glob(wave)
        files_label = glob(label)
        files_wav = random.sample(files_wav, resample)
        file_name_wave = [i.split('\\')[-1] for i in files_wav]
        file_name_json = [i.split('\\')[-1] for i in files_label]
        file_name = [i.split('.')[0] for i in file_name_wave]
        file_names_json = [name.split('.')[0] for name in file_name_json]

        matching_json_files = [files_label[i] for i, name in enumerate(file_names_json) if name in file_name]

        sampled_waveform_list.extend(files_wav)
        sampled_label_list.extend(matching_json_files)

    sampled_waveform_list = sorted(sampled_waveform_list, key=lambda x: (
        x.split('\\')[-1].split('_')[0],
        int(x.split('\\')[-1].split('_')[1].split('-')[0]),
        int(x.split('\\')[-1].split('-')[1])
    ), reverse=True)

    sampled_label_list = sorted(sampled_label_list, key=lambda x: (
        x.split('\\')[-1].split('_')[0],
        int(x.split('\\')[-1].split('_')[1].split('-')[0]),
        int(x.split('\\')[-1].split('-')[1])
    ))

    return sampled_waveform_list, sampled_label_list


def resample_audio_files(waveform_list, target_dir, target_sr=16000):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for waveform_path in waveform_list:
        # 오디오 파일 불러오기
        waveform, sr = torchaudio.load(waveform_path)
        # 재샘플링
        resampler = Resample(sr, target_sr)
        resampled_waveform = resampler(waveform)
        file_name = os.path.basename(waveform_path)
        target_file_path = os.path.join(target_dir, file_name)

        # 오디오 파일 저장
        torchaudio.save(target_file_path, resampled_waveform, target_sr)


def Json_to_txt(output_file_path, json_files_list, mode):

    if mode =='train':
        output_file_name = 'extracted_transcriptions_train.txt'
        output_file_path_txt = os.path.join(output_file_path,output_file_name)
    else:
        output_file_name = 'extracted_transcriptions_valid.txt'
        output_file_path_txt = os.path.join(output_file_path,output_file_name)

    with open(output_file_path_txt, 'w', encoding='utf-8') as out_file:
        for json_file in json_files_list:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            transcription = data.get('전사정보', {}).get('LabelText', '')
            file_name = os.path.basename(json_file)
            dot_index = file_name.rfind('.')
            if dot_index != -1:
                file_name = file_name[:dot_index]

            wav_file_path = os.path.join(mode, f'{file_name}.wav').replace('\\', '/')
            out_file.write(f'{wav_file_path}\t{transcription}\n')

    print(f"전사 정보가 파일에 저장되었습니다.")
def main():

    random.seed(1)
    train_wave_list = [r'D:\data\in_car_order\Training\waveform_train\c2h\*\*.wav',
                       r'D:\data\in_car_order\Training\waveform_train\h2c\*\*.wav',
                       r'D:\data\in_car_order\Training\waveform_train\self\*\*.wav',
                       r'D:\data\in_car_order\Training\waveform_train\sec\*\*.wav']

    train_label_list = [r'D:\data\in_car_order\Training\label_train\c2h\*\*.json',
                        r'D:\data\in_car_order\Training\label_train\h2c\*\*.json',
                        r'D:\data\in_car_order\Training\label_train\self\*\*.json',
                        r'D:\data\in_car_order\Training\label_train\sec\*\*.json']

    valid_wave_list = [r'D:\data\in_car_order\Validation\waveform_valid\c2h\*\*.wav',
                       r'D:\data\in_car_order\Validation\waveform_valid\h2c\*\*.wav',
                       r'D:\data\in_car_order\Validation\waveform_valid\sec\*\*.wav',
                       r'D:\data\in_car_order\Validation\waveform_valid\self\*\*.wav']

    valid_label_list = [r'D:\data\in_car_order\Validation\label_valid\c2h\*\*.json',
                        r'D:\data\in_car_order\Validation\label_valid\h2c\*\*.json',
                        r'D:\data\in_car_order\Validation\label_valid\sec\*\*.json',
                        r'D:\data\in_car_order\Validation\label_valid\self\*\*.json']

    train_waveform, train_label = random_sampling(train_wave_list,train_label_list,50000)
    valid_waveform, valid_label = random_sampling(valid_wave_list,valid_label_list,5000)
    test_waveform, test_label = random_sampling(valid_wave_list,valid_label_list,5000)

    target_dirs = ["./train", "./valid","./test"]

    resample_audio_files(train_waveform, target_dirs[0])
    resample_audio_files(valid_waveform, target_dirs[1])
    resample_audio_files(valid_waveform, target_dirs[2])

    Json_to_txt(target_dirs[0], train_label,'train')
    Json_to_txt(target_dirs[1], valid_label,'valid')
    Json_to_txt(target_dirs[2], valid_label, 'test')


if __name__ == "__main__":
    main()

