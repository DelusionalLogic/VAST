import json
from pathlib import (
    Path,
)

import decord
import torch
import torchaudio
from toolz.sandbox import (
    unzip,
)
from torch.utils.data import (
    Dataset,
)
from torchvision.transforms.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
)

vision_path = Path("datasets/srcdata/msrvtt/videos/")
audio_path = Path("datasets/srcdata/msrvtt/audios/")

video_sample_num = 16
audio_sample_num = 1

# Mean and stddev from clip
video_mean = [0.48145466, 0.4578275, 0.40821073] 
video_stddev  = [0.26862954, 0.26130258, 0.27577711]

frame_transform = Compose([
    Resize(224),
    CenterCrop(224),
    Normalize(video_mean, video_stddev)
])

# Mean and stddev from Bert
audio_mean =  15.41663
audio_stddev = 6.55582 

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num: #padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]

def read_video(id_):
    video_path = vision_path / (id_+".mp4")

    container = decord.VideoReader(str(video_path))
    frames_ids = list(range(len(container)))

    frames_splited = split(frames_ids, video_sample_num)

    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]

    frames = container.get_batch(sample_idx).asnumpy()
    vision_pixels = torch.from_numpy(frames.transpose(0,3,1,2)/255.0) # nX3xHxW
    vision_pixels = frame_transform(vision_pixels)

    return vision_pixels.cuda()

def read_audio(id_):
    video_path = vision_path / (id_+".mp4")

    try:
        waveform, sr = torchaudio.load(str(video_path))
    except TypeError:
        # No/Multiple audio tracks found
        return torch.zeros(audio_sample_num, 1024, 64).cuda()

    if sr != 16000:
        trans = torchaudio.transforms.Resample(sr, 16000)
        waveform = trans(waveform)

    waveform = waveform * 2 ** 15
    fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=64, sample_frequency=16000, frame_length=25, frame_shift=10)

    # normalization
    fbank = (fbank - audio_mean) / (audio_stddev * 2)
    src_length = fbank.shape[0]
    # sample 
    output_slices = []
    pad_len = max(1024 * audio_sample_num -src_length, 1024 - src_length%1024)
    fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
    total_slice_num = fbank.shape[0] // 1024
    total_slice_num = list(range(total_slice_num))
    total_slice_num = split(total_slice_num, audio_sample_num)

    sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]

    for i in sample_idx:
        cur_bank = fbank[i*1024 : (i+1)*1024]
        output_slices.append(cur_bank)

    fbank = torch.stack(output_slices,dim=0)   ### n, 1024, 128
    return fbank.cuda()

class Captions(Dataset):
    def __init__(self):
        annotation_file = Path("datasets/annotations/msrvtt/descs_ret_test.json")
        self.annos = json.load(annotation_file.open())

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]

        id_ = anno["video_id"]
        raw_captions = anno['desc']

        return {
            "ids": id_,
            "raw_captions": raw_captions,
        }

class Videos(Dataset):
    def __init__(self):
        annotation_file = Path("datasets/annotations/msrvtt/descs_ret_test.json")
        self.annos = json.load(annotation_file.open())

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]

        id_ = anno["video_id"]

        raw_subtitles = anno['subtitle']
        vision_pixels = read_video(id_)
        audio_spectrograms = read_audio(id_)

        return {
            "ids": id_,
            "vision_pixels": vision_pixels.float().cuda(),
            "audio_spectrograms": audio_spectrograms.float().cuda(),
            "raw_subtitles": raw_subtitles
        }
