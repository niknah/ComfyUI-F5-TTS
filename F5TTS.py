from pathlib import Path
import os.path
from .Install import Install
import math
import torch
import torchaudio
import hashlib
import folder_paths
import tempfile
import sys
import numpy as np
import re
import io
from omegaconf import OmegaConf
from comfy.utils import ProgressBar
import comfy
from cached_path import cached_path

# check_install will download the f5-tts if the submodule wasn't downloaded.
Install.check_install()

f5tts_path = os.path.join(Install.f5TTSPath, "src")
sys.path.insert(0, f5tts_path)
from f5_tts.model import DiT, UNetT  # noqa: E402
from f5_tts.infer.utils_infer import (  # noqa: E402
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)
sys.path.remove(f5tts_path)


class F5TTSCreate:
    voice_reg = re.compile(r"\{([^\}]+)\}")
    default_speed_type = "torch-time-stretch"
    model_names = [
        "F5v1",
        "F5",
        "F5-DE",
        "F5-ES",
        "F5-FR",
        "F5-HI",
        "F5-IT",
        "F5-JP",
        "F5-TH",
        "E2",
    ]
    vocoder_types = ["auto", "vocos", "bigvgan"]
    tooltip_seed = "Seed. -1 = random"
    tooltip_speed = "Speed. >1.0 slower. <1.0 faster"
    tooltip_audio = "5-15 seconds of audio"

    def get_model_names():
        model_names = F5TTSCreate.model_names[:]
        models_path = folder_paths.get_folder_paths("checkpoints")
        for model_path in models_path:
            f5_model_path = os.path.join(model_path, 'F5-TTS')
            if os.path.isdir(f5_model_path):
                for file in os.listdir(f5_model_path):
                    p = Path(file)
                    if (
                        p.suffix in folder_paths.supported_pt_extensions
                        and os.path.isfile(os.path.join(f5_model_path, file))
                    ):
                        txtFile = F5TTSCreate.get_txt_file_path(
                            os.path.join(f5_model_path, file)
                        )

                        if (
                            os.path.isfile(txtFile)
                        ):
                            model_names.append("model://"+file)
        return model_names

    @staticmethod
    def get_txt_file_path(file):
        p = Path(file)
        return os.path.join(os.path.dirname(file), p.stem + ".txt")

    def is_voice_name(self, word):
        return self.voice_reg.match(word.strip())

    def get_voice_names(self, chunks):
        voice_names = {}
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice_names[match[1]] = True
        return voice_names

    def split_text(self, speech):
        reg1 = r"(?=\{[^\}]+\})"
        return re.split(reg1, speech)

    @staticmethod
    def load_voice(ref_audio, ref_text):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}

        main_voice["ref_audio"], main_voice["ref_text"] = \
            preprocess_ref_audio_text(  # noqa: E501
                ref_audio, ref_text
            )
        return main_voice

    def get_model_funcs(self):  # noqa: E501
        return {
            "F5": self.load_f5_model,
            "F5v1": {
                "model": "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",  # noqa: E501
                "vocab": "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",  # noqa: E501
                "model_type": "F5TTS_v1_Base",
            },
            "F5-HI": {
                "model": "hf://ShriAishu/hindiSpeech/model.safetensors",
                "vocab": "hf://ShriAishu/hindiSpeech/checkpoints/vocab.txt",
            },
            "F5-JP": {
                "model": "hf://Jmica/F5TTS/JA_8500000/model_8499660.pt",
                "vocab": "hf://Jmica/F5TTS/JA_8500000/vocab_updated.txt",
            },
            "F5-FR": {
                "model": "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_1374000.pt",  # noqa: E501
                "vocab": "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt",  # noqa: E501
            },
            "F5-DE": {
                "model": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_420000.safetensors",  # noqa: E501
                "vocab": "hf://aihpi/F5-TTS-German/vocab.txt",
            },
            "F5-IT": {
                "model": "hf://alien79/F5-TTS-italian/model_159600.safetensors",  # noqa: E501
                "vocab": "hf://alien79/F5-TTS-italian/vocab.txt",
            },
            "F5-ES": {
                "model": "hf://jpgallegoar/F5-Spanish/model_1200000.safetensors",  # noqa: E501
                "vocab": "hf://jpgallegoar/F5-Spanish/vocab.txt",
            },
            "F5-TH": {
                "model": "hf://VIZINTZOR/F5-TTS-THAI/model_600000.pt",  # noqa: E501
                "vocab": "hf://VIZINTZOR/F5-TTS-THAI/vocab.txt",
            },
            "E2": self.load_e2_model,
        }

    def get_vocoder(self, vocoder_name):
        if vocoder_name == "vocos":
            os.path.join(Install.f5TTSPath, "checkpoints/vocos-mel-24khz")
        elif vocoder_name == "bigvgan":
            os.path.join(Install.f5TTSPath, "checkpoints/bigvgan_v2_24khz_100band_256x")  # noqa: E501

    def load_vocoder(self,  vocoder_name):
        sys.path.insert(0, f5tts_path)
        vocoder = load_vocoder(vocoder_name=vocoder_name)
        sys.path.remove(f5tts_path)
        return vocoder

    def get_model_config(self, model_type):
        config_path = F5TTSCreate.get_config_path()
        return OmegaConf.load(
            os.path.join(config_path, model_type + ".yaml")
            ).model

    def load_model(self, model, vocoder_name, model_type):
        model_funcs = self.get_model_funcs()
        if model in model_funcs:
            if vocoder_name == 'auto':
                vocoder_name = 'vocos'
            func = model_funcs[model]
            if isinstance(func, dict):
                model_cfg = None
                if "model_type" in func:
                    model_config = self.get_model_config(func["model_type"])
                    model_cfg = model_config.arch

                return self.load_f5_model_url(
                    func["model"],
                    vocoder_name,
                    func["vocab"],
                    model_cfg=model_cfg
                )
            return func(vocoder_name)
        else:
            model_cfg = self.get_model_config(model_type)
            if vocoder_name == 'auto':
                vocoder_name = model_cfg.mel_spec.mel_spec_type

            return self.load_f5_model_url(
                model, vocoder_name,
                model_cfg=model_cfg.arch
                )

    def get_vocab_file(self):
        return os.path.join(
            Install.f5TTSPath, "data/Emilia_ZH_EN_pinyin/vocab.txt"
            )

    def load_e2_model(self, vocoder):
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
            )
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))  # noqa: E501
        vocab_file = self.get_vocab_file()
        vocoder_name = "vocos"
        ema_model = load_model(
            model_cls, model_cfg,
            ckpt_file, vocab_file=vocab_file,
            mel_spec_type=vocoder_name,
            )
        vocoder = self.load_vocoder(vocoder_name)
        return (ema_model, vocoder, vocoder_name)

    def load_f5_model(self, vocoder):
        repo_name = "F5-TTS"
        extension = "safetensors"
        if vocoder == "bigvgan":
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            extension = "pt"
        else:
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
        return self.load_f5_model_url(
            f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.{extension}",  # noqa: E501
            vocoder,
        )

    def cached_path(self, url):
        if url.startswith("model:"):
            path = re.sub("^model:/*", "", url)
            models_path = folder_paths.get_folder_paths("checkpoints")
            for model_path in models_path:
                f5_model_path = os.path.join(model_path, 'F5-TTS')
                model_file = os.path.join(f5_model_path, path)
                if os.path.isfile(model_file):
                    return model_file
            raise FileNotFoundError("No model found: " + url)
            return None
        return str(cached_path(url))

    def load_f5_model_hi_old(self, vocoder):
        model_cfg = dict(
            dim=768,
            depth=18,
            heads=12,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
            checkpoint_activations=False,
            )
        return self.load_f5_model_url(
            "hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors",
            "vocos",
            "hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt",
            model_cfg=model_cfg,
            )

    def load_f5_model_url(
        self, url, vocoder_name, vocab_url=None, model_cfg=None
    ):
        vocoder = self.load_vocoder(vocoder_name)
        model_cls = DiT
        if model_cfg is None:
            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                text_mask_padding=False,
                conv_layers=4,
                pe_attn_head=1,
                )

        ckpt_file = str(self.cached_path(url))

        if vocab_url is None:
            if url.startswith("model:"):
                vocab_file = F5TTSCreate.get_txt_file_path(ckpt_file)
            else:
                vocab_file = self.get_vocab_file()
        else:
            vocab_file = str(self.cached_path(vocab_url))
        ema_model = load_model(
            model_cls, model_cfg,
            ckpt_file, vocab_file=vocab_file,
            mel_spec_type=vocoder_name,
            )
        return (ema_model, vocoder, vocoder_name)

    def generate_audio(
        self, voices, model_obj, chunks, seed, vocoder, mel_spec_type,
        infer_args={}
    ):
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.random.seed()

        frame_rate = 44100
        generated_audio_segments = []
        pbar = ProgressBar(len(chunks))
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = F5TTSCreate.voice_reg.sub("", text)
            gen_text = text.strip()
            if gen_text == "":
                print(f"No text for {voice}, skip")
                continue
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            print("text:"+text)
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model_obj,
                vocoder=vocoder, mel_spec_type=mel_spec_type,
                device=comfy.model_management.get_torch_device(),
                **infer_args
                )
            generated_audio_segments.append(audio)
            frame_rate = final_sample_rate
            pbar.update(1)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

        # waveform, sample_rate = torchaudio.load(wave_file.name)
        waveform = torch.from_numpy(final_wave).unsqueeze(0)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": frame_rate
            }
        # os.unlink(wave_file.name)
        return audio

    def time_shift(self, audio, speed, speed_type):
        if speed == 1:
            return audio
        elif speed_type == "TDHS":
            return self.time_shift_audiostretchy(audio, speed)
        elif speed_type == "torch-time-stretch":
            return self.time_shift_torch_ts(audio, speed)
        return audio

    def create(
        self, voices, chunks, seed=-1, model="F5",
        vocoder_name="vocos",
        model_type='F5TTS_Base', infer_args={}
    ):
        (
            model_obj,
            vocoder,
            mel_spec_type
        ) = self.load_model(model, vocoder_name, model_type)
        return self.generate_audio(
            voices,
            model_obj,
            chunks, seed,
            vocoder, mel_spec_type=mel_spec_type,
            infer_args=infer_args,
        )

    def time_shift_audiostretchy(self, audio, speed):
        from audiostretchy.stretch import AudioStretch

        rate = audio['sample_rate']
        waveform = audio['waveform']

        new_waveforms = []
        for channel in range(0, waveform.shape[0]):
            ta_audio16 = waveform[0][channel] * 32768

            audio_stretch = AudioStretch()
            audio_stretch.samples = audio_stretch.in_samples = \
                ta_audio16.numpy().astype('int16')
            audio_stretch.nchannels = 1
            audio_stretch.sampwidth = 2
            audio_stretch.framerate = rate
            audio_stretch.nframes = waveform.shape[2]
            audio_stretch.stretch(ratio=speed)

            new_waveforms.append(torch.from_numpy(audio_stretch.samples))
        new_waveform = torch.stack(new_waveforms)
        new_waveform = torch.stack([new_waveform])

        return {"waveform": new_waveform, "sample_rate": rate}

    def time_shift_torch_ts(self, audio, speed):
        import torch_time_stretch
        rate = audio['sample_rate']
        waveform = audio['waveform']

        new_waveform = torch_time_stretch.time_stretch(
            waveform,
            torch_time_stretch.Fraction(math.floor(speed*100), 100),
            rate
        )

        return {"waveform": new_waveform, "sample_rate": rate}

    @staticmethod
    def load_voice_from_file(sample):
        input_dir = folder_paths.get_input_directory()
        txt_file = os.path.join(
            input_dir,
            F5TTSCreate.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r', encoding='utf-8') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)
        print(f"audio_text {audio_text}")
        return F5TTSCreate.load_voice(audio_path, audio_text)

    @staticmethod
    def load_voices_from_files(sample, voice_names):
        voices = {}
        p = Path(sample)
        for voice_name in voice_names:
            if voice_name == "main":
                continue
            sample_file = os.path.join(
                os.path.dirname(sample),
                "{stem}.{voice_name}{suffix}".format(
                    stem=p.stem,
                    voice_name=voice_name,
                    suffix=p.suffix
                    )
                )
            print("voice:"+voice_name+","+sample_file+','+sample)
            voices[voice_name] = F5TTSCreate.load_voice_from_file(sample_file)
        return voices

    @staticmethod
    def remove_wave_file(wave_file_name):
        if wave_file_name is not None:
            try:
                os.unlink(wave_file_name)
            except Exception as e:
                print("F5TTS: Cannot remove? "+wave_file_name)
                print(e)

    @staticmethod
    def load_voice_from_input(sample_audio, sample_text):
        wave_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
            )
        wave_file_name = wave_file.name
        wave_file.close()

        hasAudio = False
        for (batch_number, waveform) in enumerate(
            sample_audio["waveform"].cpu()
        ):
            buff = io.BytesIO()
            torchaudio.save(
                buff, waveform, sample_audio["sample_rate"], format="WAV"
                )
            with open(wave_file_name, 'wb') as f:
                f.write(buff.getbuffer())
            hasAudio = True
            break
        if not hasAudio:
            raise FileNotFoundError("No audio input")
        r = F5TTSCreate.load_voice(wave_file_name, sample_text)
        return (r, wave_file_name)

    @staticmethod
    def get_config_path():
        return os.path.join(
            Install.f5TTSPath,
            'src/f5_tts/configs'
            )

    @staticmethod
    def get_configs():
        config_path = F5TTSCreate.get_config_path()
        configs = []
        for file in os.listdir(config_path):
            p = Path(file)
            if file.endswith('.yaml'):
                configs.append(p.stem)
        return configs


class F5TTSAudioInputs:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        model_names = F5TTSCreate.get_model_names()
        model_types = F5TTSCreate.get_configs()
        return {
            "required": {
                "sample_audio": ("AUDIO",{
                    "tooltip": F5TTSCreate.tooltip_audio,
                }),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (model_names, {
                    "default": "F5v1",
                }),
                "vocoder": (F5TTSCreate.vocoder_types, {
                    "tooltip": "Auto will be set by model_type. Most models are usually vocos.",  # noqa: E501
                    "default": "auto",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "tooltip": F5TTSCreate.tooltip_speed,
                }),
                "model_type": (model_types, {
                    "tooltip": "Type of model",
                    "default": 'F5TTS_Base',
                }),
            },
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"
    DESCRIPTION = "From one audio input.  (Does not support multi voice)"

    def create(
        self,
        sample_audio, sample_text,
        speech, seed=-1, model="F5", vocoder="vocos",
        speed=1,
        model_type=None,
    ):

        wave_file_name = None
        try:
            (main_voice, wave_file_name) = F5TTSCreate.load_voice_from_input(
                sample_audio, sample_text
            )

            f5ttsCreate = F5TTSCreate()

            voices = {}
            chunks = f5ttsCreate.split_text(speech)
            voices['main'] = main_voice

            audio = f5ttsCreate.create(
                voices, chunks, seed, model, vocoder,
                model_type
            )
            audio = f5ttsCreate.time_shift(
                audio, speed, F5TTSCreate.default_speed_type
                )
        finally:
            if wave_file_name is not None:
                F5TTSCreate.remove_wave_file(wave_file_name)
        return (audio, )

    @classmethod
    def IS_CHANGED(
        s, sample_audio, sample_text,
        speech, seed, model, vocoder, speed,
        model_type
    ):
        m = hashlib.sha256()
        m.update(sample_text)
        m.update(sample_audio)
        m.update(speech)
        m.update(seed)
        m.update(model)
        m.update(vocoder)
        m.update(speed)
        m.update(model_type)
        return m.digest().hex()


class F5TTSAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        input_dirs = [
                "",
                'audio',
                'F5-TTS',
        ]
        files = []
        for dir_short in input_dirs:
            d = os.path.join(input_dir, dir_short)
            if os.path.exists(d):
                dir_files = folder_paths.filter_files_content_types(
                    os.listdir(d), ["audio", "video"]
                    )
                dir_files = [os.path.join(dir_short, s) for s in dir_files]
                files.extend(dir_files)
        filesWithTxt = []
        for file in files:
            txtFile = F5TTSCreate.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        filesWithTxt = sorted(filesWithTxt)

        model_names = F5TTSCreate.get_model_names()
        model_types = F5TTSCreate.get_configs()

        return {
            "required": {
                "sample": (filesWithTxt, {"audio_upload": True}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (model_names,),
                "vocoder": (F5TTSCreate.vocoder_types, {
                    "tooltip": "Most models are usally vocos",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": F5TTSCreate.tooltip_speed,
                }),
                "model_type": (model_types, {
                    "tooltip": "Type of model",
                    "default": 'F5TTS_Base',
                }),
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"
    DESCRIPTION = "Put audio + txt into inputs/F5-TTS. (Supports Multi Voice)"

    def create(
        self,
        sample, speech, seed=-1, model="F5", vocoder="vocos",
        speed=1,
        model_type=None,
    ):
        # vocoder = "vocos"
        # Install.check_install()
        main_voice = F5TTSCreate.load_voice_from_file(sample)

        f5ttsCreate = F5TTSCreate()

        chunks = f5ttsCreate.split_text(speech)
        voice_names = f5ttsCreate.get_voice_names(chunks)
        voices = F5TTSCreate.load_voices_from_files(sample, voice_names)
        voices['main'] = main_voice

        audio = f5ttsCreate.create(
            voices, chunks, seed, model, vocoder,
            model_type
            )

        audio = f5ttsCreate.time_shift(
            audio, speed, F5TTSCreate.default_speed_type
            )
        return (audio, )

    @classmethod
    def IS_CHANGED(
        s,
        sample, speech, seed, model, vocoder, speed,
        model_type
    ):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        audio_txt_path = F5TTSCreate.get_txt_file_path(audio_path)
        last_modified_timestamp = os.path.getmtime(audio_path)
        txt_last_modified_timestamp = os.path.getmtime(audio_txt_path)
        m.update(audio_path)
        m.update(str(last_modified_timestamp))
        m.update(str(txt_last_modified_timestamp))
        m.update(speech)
        m.update(seed)
        m.update(model)
        m.update(vocoder)
        m.update(speed)
        m.update(model_type)
        return m.digest().hex()


class F5TTSAudioAdvanced:
    default_sample_text = "Text of sample_audio"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        input_dirs = [
                "",
                'audio',
                'F5-TTS',
        ]
        files = []
        for dir_short in input_dirs:
            d = os.path.join(input_dir, dir_short)
            if os.path.exists(d):
                dir_files = folder_paths.filter_files_content_types(
                    os.listdir(d), ["audio", "video"]
                    )
                dir_files = [os.path.join(dir_short, s) for s in dir_files]
                files.extend(dir_files)
        filesWithTxt = []
        for file in files:
            txtFile = F5TTSCreate.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        filesWithTxt = sorted(filesWithTxt)

        model_names = F5TTSCreate.get_model_names()
        model_types = F5TTSCreate.get_configs()

        return {
            "required": {
                "sample": (filesWithTxt, {
                    "audio_upload": True,
                    "tooltip": F5TTSCreate.tooltip_audio,
                }),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (model_names,),
                "vocoder": (F5TTSCreate.vocoder_types, {
                    "tooltip": "Most models are usally vocos",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": F5TTSCreate.tooltip_seed
                }),
                "model_type": (model_types, {
                    "tooltip": "Type of model",
                    "default": 'F5TTS_Base',
                }),
            },
            "optional": {
                "sample_audio": ("AUDIO", {
                    "tooltip": "When this is connected, sample is ignored.  Also put the words into sample_text",  # noqa: E501
                }),
                "sample_text": ("STRING", {
                    "default": F5TTSAudioAdvanced.default_sample_text,
                    "multiline": True,
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1,
                    "tooltip": "Target output speech loudness normalization value",  # noqa: E501
                    "step": 0.01,
                }),
                "cross_fade_duration": ("FLOAT", {
                    "default": 0.15,
                    "tooltip": "Duration of cross-fade between audio segments in seconds",  # noqa: E501
                    "step": 0.01,
                }),
                "nfe_step": ("INT", {
                    "default": 32,
                    "tooltip": "The number of function evaluation (denoising steps)",  # noqa: E501
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2,
                    "tooltip": "Classifier-free guidance strength",
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1,
                    "tooltip": "Sway Sampling coefficient",
                    "min": -10,
                    "step": 0.001,
                }),
                "speed_type": (["torch-time-stretch", "F5TTS", "TDHS"], {
                    "default": "torch-time-stretch",
                    "tooltip": "TDHS - Time-domain harmonic scaling. torch-time-stretch - torchaudio.transforms.TimeStretch. F5TTS's default time stretch. ",  # noqa: E501
                }),
                "fix_duration": ("FLOAT", {
                    "default": -1,
                    "tooltip": "Fix the total duration (ref and gen audios) in second. -1 = disable",  # noqa: E501
                    "min": -1,
                    "step": 0.01,
                }),
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"
    DESCRIPTION = "Advanced with extra options from F5-TTS."

    def create(
        self,
        sample, speech, seed=-1, model="F5", vocoder="vocos",
        speed=1,
        model_type=None,
        sample_audio=None,
        sample_text="",
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2,
        sway_sampling_coef=-1,
        speed_type="torch-time-stretch",
        fix_duration=-1,
    ):
        wave_file_name = None
        try:
            f5ttsCreate = F5TTSCreate()
            voices = {}

            if sample_audio is not None:
                if sample_text == F5TTSAudioAdvanced.default_sample_text:
                    raise Exception(
                        "Must change sample_text to what was said in the audio input."  # noqa: E501
                    )
                (
                    main_voice, wave_file_name
                ) = F5TTSCreate.load_voice_from_input(
                        sample_audio, sample_text
                    )
                chunks = f5ttsCreate.split_text(speech)
            else:
                main_voice = F5TTSCreate.load_voice_from_file(sample)
                chunks = f5ttsCreate.split_text(speech)
                voice_names = f5ttsCreate.get_voice_names(chunks)
                voices = F5TTSCreate.load_voices_from_files(
                    sample, voice_names
                )
            voices['main'] = main_voice
            infer_args = {}
            infer_args['target_rms'] = target_rms
            infer_args['cross_fade_duration'] = cross_fade_duration
            infer_args['nfe_step'] = nfe_step
            infer_args['cfg_strength'] = cfg_strength
            infer_args['sway_sampling_coef'] = sway_sampling_coef
            if (speed_type == "F5TTS" and speed != 1):
                infer_args['speed'] = 1 / speed
            if (fix_duration >= 0):
                infer_args['fix_duration'] = fix_duration

            audio = f5ttsCreate.create(
                voices, chunks, seed, model, vocoder,
                model_type, infer_args
                )
            audio = f5ttsCreate.time_shift(audio, speed, speed_type)
        finally:
            if wave_file_name is not None:
                F5TTSCreate.remove_wave_file(wave_file_name)

        return (audio, )

    @classmethod
    def IS_CHANGED(
        s,
        sample, speech, seed, model, vocoder, speed,
        model_type,
        sample_audio,
        sample_text,
        target_rms,
        cross_fade_duration,
        nfe_step,
        cfg_strength,
        sway_sampling_coef,
        speed_type,
        fix_duration,
    ):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        audio_txt_path = F5TTSCreate.get_txt_file_path(audio_path)
        last_modified_timestamp = os.path.getmtime(audio_path)
        txt_last_modified_timestamp = os.path.getmtime(audio_txt_path)
        m.update(audio_path)
        m.update(str(last_modified_timestamp))
        m.update(str(txt_last_modified_timestamp))
        m.update(speech)
        m.update(seed)
        m.update(model)
        m.update(vocoder)
        m.update(speed)
        m.update(model_type)
        m.update(sample_audio)
        m.update(sample_text)
        m.update(target_rms)
        m.update(cross_fade_duration)
        m.update(nfe_step)
        m.update(cfg_strength)
        m.update(sway_sampling_coef)
        m.update(speed_type)
        m.update(fix_duration)
        return m.digest().hex()
