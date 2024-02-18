# ASR_with_openspeech

This project utilizes the vehicle command dataset from AI Hub. You can access the dataset [here](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=112).
[openspeech](https://github.com/openspeech-team/openspeech)의 여러가지 모델들을 이용해서, 상기의 데이터 셋을 이용한 한국어 음성인식 모델을 구성하는 것이 목표
같은 프로젝트를 한분과 진행상황 공유 하였으나, 서로 코드 수정 및 학습은 개인적으로 진행하였습니다.

참고한 레퍼지토리
https://github.com/openspeech-team/openspeech/issues/81 : issue에 있는 customdataset 구성 관련문서
https://github.com/alsrb0607/KoreanSTT : openspeech 구버전 같은 kospeech로 유사한 프로젝트 한것을 참고

모델선정 및 진행과정
1월~2월 중순까지 진행하였고,
[Criterion](https://github.com/openspeech-team/openspeech/wiki/Criterion-suitable-for-the-model#criterion)의 loss 기준을 참고하여 10여개의 모델을 scratch부터 학습을 시도, 
CTC:  loss는 잘내려가나 wer과 cer이 어느 지점에서 수렴하여 내려가지 않음.
joint_ctc_cross_entropy:  wer cer이 수렴하여 내려가지 않음
transducer : 호환오류로 학습이 제한되는 불상사가 발생
label_smoothed_cross_entropy: crossentropy와 큰 차이가 없다고 생각하여, 제외하고 진행하였으며, loss가 시작부터 1-e3 급으로 떠서 신뢰성이 없었음
crossentropy : 여기서 최종적으로 listen_attend_spell_with_multi_head과 conformer_lstm을 사용하여 학습진행
-->표로 만들어줘 GPT야


전처리
tokenizer는 ksponspeech의 tokenizer를 custom 데이터만 받아올수 있게 설정하였으며 character를 사용
vehicle command dataset에서 train valid test 셋 각각 (20만,2만,2만)으로 설정하여 학습진행
manifest 파일을 만들기 위해선, 각 waveform의 위치와 해당 waveform의 transcription이 필요.
따라서, resampling을 따로 진행하여 waveform{/t}transcription으로 txt파일을 만들어서 넣었음

예시
test/EA_0625-1507-04-01-LSL-F-05-B.wav	조금 긴장되네 편안하고 듣기 편한 노래 틀어 줘.

학습진행 및 결과 
conformer_lstm 및 listen_attend_spell_with_multi_head를 이용해 각각 학습 
conformer_lstm는 3090 2대, listen_attend_spell_with_multi_head는 3090 1대로 학습하였음

train 명령어(conformer_lstm의 경우)
python3 ./openspeech_cli/hydra_train.py \
dataset=ksponspeech \
dataset.dataset_path=/home/user/PycharmProjects/model3/openspeech/in_car_command/ \
dataset.manifest_file_path=/home/user/PycharmProjects/model3/openspeech/in_car_command/in_car_command_manifest.txt \
tokenizer=kspon_character \
tokenizer.vocab_path=/home/user/PycharmProjects/model3/openspeech/in_car_command/in_command_car.csv  \
model=conformer_lstm \
audio=melspectrogram \
lr_scheduler=warmup_reduce_lr_on_plateau \
trainer=gpu \
criterion=cross_entropy

![las_reslut](https://github.com/Angeriod/ASR_with_openspeech/assets/97516571/c749cb55-8ee7-40c4-8c44-99475c2fd556)

예측 및 최종성능 평가
총 2만개 데이터로 진행
eval 명령어
python3 ./openspeech_cli/hydra_eval.py \
audio=melspectrogram \
eval.dataset_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/ \
eval.checkpoint_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/completed_model/conformer_lstm_in_car_command/57_360000.ckpt \
eval.manifest_file_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/in_car_command_test_manifest.txt \
eval.result_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/conformer_lstm_test_result_57epoch_beam4.txt \
model=conformer_lstm \
tokenizer=kspon_character \
tokenizer.vocab_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/in_command_car.csv 
여기서 txt 파일에 waveform{/t}transcription을 test만 넣어서 eval.dataset_path에서 가져올수 있게함

결과
conformer_lstm
Word Error Rate: 0.39726612194018257, Character Error Rate: 0.334399943365839
listen_attend_spell_with_multi_head는
Word Error Rate: 0.20948063147289603, Character Error Rate: 0.14763109923366902

listen_attend_spell_with_multi_head의 경우

잘된예시
최근 통화 목록에서 김민수 찾아서 연결해 줘.
최근 통화 목록에서 김민수 찾아서 연결해 줘.

잘 안된 예시
차선 좌로 한 칸 옮겨서 운전해.
차선 지금처럼 다음 일도 낮게 설정해.

결론
train의 데이터수가 적어서 test가 별로 였던것 같았습니다. conformer_lstm은 20만개부터 val_loss가 발산하지 않았고 warmup을 40000스텝으로 늘려서 진행하여 loss를 유지하여 train상에서 cer과 wer이 매우 양호했으나, test에서 굉장이 별로 였습니다
listen_attend_spell_with_multi_head도 데이터 1만개일때는 val_loss가 발산, 5만개일때는 수렴 하지만 test wer cer은 별로 좋지않았습니다.
resoucre의 한계가 컸던 프로젝트였으나, 나름 만족할만한 결과를 낳아서 좋았습니다.
