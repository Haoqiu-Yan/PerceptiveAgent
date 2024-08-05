"""
CUDA_VISIBLE_DEVICES=1 python code/synthesis/unit_to_vocoder_test.py
"""

import torch
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder

# Available models
EXPRESSO_MODELS = [
    ("hubert-base-ls960-layer-9", "kmeans", 500),
    ("hubert-base-ls960-layer-9", "kmeans-expresso", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans-expresso", 2000),
]

# Try one model
dense_model, quantizer_model, vocab = EXPRESSO_MODELS[1]

# Load speech encoder and vocoder
encoder = SpeechEncoder.by_name(
    dense_model_name = dense_model,
    quantizer_model_name = quantizer_model,
    vocab_size = vocab,
    deduplicate = False, # False if the vocoder doesn't support duration prediction
).cuda()

vocoder = CodeHiFiGANVocoder.by_name(
    dense_model_name = dense_model,
    quantizer_model_name = quantizer_model,
    vocab_size = vocab,
    speaker_meta = True,
    style_meta = True
).cuda()
speakers = vocoder.speakers # ['ex01', 'ex02', 'ex03', 'ex04', 'lj', 'vctk_p225', ...]
styles = vocoder.styles # ['read-default', 'read-happy', 'read-sad', 'read-whisper', ...]

# # Load the audio
input_file = "datasets/expresso/merge_audio_48khz/ex04_happy_00194.wav"
waveform, sample_rate = torchaudio.load(input_file)

# # Convert it to (duplicated) units
encoded = encoder(waveform.cuda())
units = encoded["units"] # torch.Tensor([17, 17, 17, 17, 296, 296,...])

# audio: datasets/expresso/merge_audio_16khz/ex04_happy_00194.wav
# units = torch.LongTensor([123,123,123,1761,375,375,1556,400,171,439,439,982,887,391,391,391,391,391,391,391,1090,144,1230,1230,1230,363,363,607,1267,1267,40,938,151,151,824,1352,2,934,1249,1249,1249,1483,744,1556,744,744,396,744,744,744,1476,114,1082,1219,555,904,1266,1266,1266,1266,1266,796,516,345,654,654,1217,16,366,1205,473,1401,203,122,228,1716,1696,1061,1916,205,1223,1223,1223,1223,1223,1721,357,1208,1494,1494,1494,1494,256,261,642,422,422,422,396,396,396,105,105,1018,6]).cuda()

# Convert units back to audio
audio = vocoder(
    units,
    speaker_id=speakers.index('ex04'),
    style_id=styles.index('read-happy'),
) # torch.Tensor([-9.9573e-04, -1.7003e-04, -6.8756e-05,...])

# 遇到了问题: must be 2D
# torchaudio.save('code/synthesis/ex04_happy_00194_gen.wav', audio[0].cpu(), sample_rate=48000)
torchaudio.save('code/synthesis/ex04_happy_00194_gen_16khz.wav', audio.cpu().float().unsqueeze(0), sample_rate=vocoder.output_sample_rate)
