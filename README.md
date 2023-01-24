# Pitch Controllable DDSP Vocoders
<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>
The repository is a collection of relatively high fidelity, fast, easy trained pitch controllable ddsp vocoders modified from the below repositorys:

https://github.com/magenta/ddsp

https://github.com/YatingMusic/ddsp-singing-vocoders

## 1. Installing the dependencies
We recommend first installing PyTorch from the [**official website**](https://pytorch.org/), then run:
```bash
pip install -r requirements.txt 
```
## 2. Preprocessing
Put all the training dataset (.wav format audio clips) in the below directory:
`data/train/audio`.
Put all the validation dataset (.wav format audio clips) in the below directory:
`data/val/audio`.
Then run
```bash
python preprocess.py -c configs/full.yaml
```
for a model of hybrid additive synthesis and subtractive synthesis, or run
```bash
python preprocess.py -c configs/sins.yaml
```
for a model of additive synthesis only, or run
```bash
python preprocess.py -c configs/sawsub.yaml
```
for a model of substractive synthesis only.

You can modify the configuration file `config/<model_name>.yaml` before preprocessing. The default configuration during training is 44.1khz sampling rate audio for about a few hours and GTX1660 graphics card.

NOTE 1: Please keep the sampling rate of all audio clips consistent with the sampling rate in the yaml configuration file ! If it is not consistent, the program can be executed safely, but the resampling during the training process will be very slow.

NOTE 2: The total number of audio clips is recommended to be more than 1000, especially long audio clip can be cut into short segments, which will speed up the training, but the duration of all audio clips should not be less than 2 seconds.

## 3. Training
```bash
# train a full model as an example
python train.py -c configs/full.yaml
```
The command line for training other models is similar.

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## 4. Visualization
```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```

## 5. Copy-synthesising or pitch-shifting test 
```bash
# Copy-synthesising test
# wav -> mel, f0 -> wav
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)>
```
```bash
# Pitch-shifting test
# wav -> mel, f0 -> mel (unchaned), f0 (shifted) -> wav
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <key(semitones)>
```

## 6. Some suggestions for the model choice
It is recommended to try the "Full" model first, which generally has a low multi-scaled-stft loss and relatively good quality when applying a pitch shift.

However, this loss sometimes cannot reflect the subjective sense of hearing.

If the "Full" model does not work well, it is recommended to switch to the "Sins" model.

The "Sins" model works also well when applying copy synthesis, but it changes the formant when applying a pitch shift, which changes the timbre.

The "SawSub" model is not recommended due to artifacts in unvoiced phonemes, although it probably has the best formant invariance in pitch-shifting cases.

## 7. Comments on the sound quality
For the seen speaker, the sound quality of a well-trained ddsp vocoder will be better than that of the world vocoder or griffin-lim vocoder, and it can also compete with the gan-based vocoder when the total amount of data is relatively small. But for a large amount of data, the upper limit of sound quality will be lower than that of generative model-based vocoders.

For the unseen speaker, the performance may be unsatisfactory.





