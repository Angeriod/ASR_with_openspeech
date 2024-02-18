# ASR_with_openspeech

This project focuses on constructing a Korean speech recognition model using the vehicle command dataset from AI Hub, accessible [here](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=112). Utilizing various models from [openspeech](https://github.com/openspeech-team/openspeech), the goal is to explore and enhance speech recognition capabilities. Although this project was shared with a colleague, code modifications and training were conducted individually.

## Reference Repositories

- Issues on custom dataset configuration in openspeech: [openspeech issue #81](https://github.com/openspeech-team/openspeech/issues/81)
- Similar project using an earlier version of openspeech, kospeech: [KoreanSTT by alsrb0607](https://github.com/alsrb0607/KoreanSTT)

## Model Selection and Progress

From January to mid-February, based on the loss criteria from [openspeech Criterion](https://github.com/openspeech-team/openspeech/wiki/Criterion-suitable-for-the-model#criterion), several models were trained from scratch:

| Loss Type                  | Outcome |
|----------------------------|---------|
| CTC                        | Loss decreased, but WER and CER plateaued |
| Joint CTC Cross Entropy    | WER and CER did not improve |
| Transducer                 | Encountered compatibility issues, limiting training |
| Label Smoothed Cross Entropy | Excluded due to similarity to cross-entropy and initial high loss |
| Crossentropy               | Final selection: trained using Listen, Attend and Spell with Multi-Head and Conformer LSTM models |

## Preprocessing

- The tokenizer from ksponspeech was customized to work with our dataset, using character-level encoding. The vehicle command dataset was split into training, validation, and testing sets with 200,000, 20,000, and 20,000 entries, respectively. Each entry required waveform location and corresponding transcription, processed into a txt file format like `waveform{\t}transcription` for manifest file generation.
- The dataset encompassed four distinct domains: `sec`, `h2c`, `c2h`, and `self`, each representing a unique area of vehicle commands. To ensure a balanced representation from each domain, we employed a random sampling strategy to construct the training, validation, and test sets. 

## Training and Results

Models used include Conformer LSTM and Listen, Attend and Spell with Multi-Head, trained on NVIDIA RTX 3090 GPUs.

### Training Command for Conformer LSTM

```shell
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
```
## Evaluation and Results

A comprehensive evaluation was conducted on a test set comprising 20,000 entries to gauge the performance of the trained models.

### Evaluation Process

The evaluation was executed using the following command, tailored to assess the model's ability to accurately transcribe speech into text:

### Evaluation Command for Conformer LSTM
```shell
python3 ./openspeech_cli/hydra_eval.py \
  audio=melspectrogram \
  eval.dataset_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/ \
  eval.checkpoint_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/completed_model/conformer_lstm_in_car_command/57_360000.ckpt \
  eval.manifest_file_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/in_car_command_test_manifest.txt \
  eval.result_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/conformer_lstm_test_result_57epoch_beam4.txt \
  model=conformer_lstm \
  tokenizer=kspon_character \
  tokenizer.vocab_path=/home/user/PycharmProjects/BH_flask/model3/openspeech/in_car_command/in_command_car.csv
```
## Performance Metrics

The models demonstrated varying levels of accuracy, with the results detailed below:

- **Conformer LSTM**:
  - Word Error Rate (WER): 39.726%
  - Character Error Rate (CER): 33.439%

- **Listen, Attend and Spell with Multi-Head**:
  - WER: 20.948%
  - CER: 14.763%

### Success and Failure Examples from LAS-MH

**Successful Prediction**:

- **Input**: "최근 통화 목록에서 김민수 찾아서 연결해 줘."
- **Prediction**: "최근 통화 목록에서 김민수 찾아서 연결해 줘."

**Unsuccessful Prediction**:

- **Input**: "차선 좌로 한 칸 옮겨서 운전해."
- **Prediction**: "차선 지금처럼 다음 일도 낮게 설정해."

## Conclusion and Reflections

The journey through this project highlighted the critical role of dataset size in training effective speech recognition models. The Conformer LSTM model, despite promising training performance, exhibited limitations in test scenarios, possibly due to an inadequate volume of training data. The Listen, Attend and Spell with Multi-Head model also faced challenges, Performance was not good on too few datasets.


