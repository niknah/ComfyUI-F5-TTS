
ComfyUI node to make text to speech audio with your own voice.

Using F5-TTS https://github.com/SWivid/F5-TTS


### Instructions

* Install [ffmpeg-full-shared](https://ffmpeg.org/download.html).  If you are using conda, use `conda install ffmpeg`

* Put in ComfyUI's `input` folder a .wav file of an audio of the voice you'd like to use, remove any background music, noise.
* And a .txt file of the same name with what was said.
* Press refresh to see it in the node

* `input/F5-TTS` `input/audio` folders will also work.

You can use the examples here...
* [Examples voices](example_workflows/)
* [Simple workflow](example_workflows/simple_ComfyUI_F5TTS_workflow.json)
* [Workflow with input audio only, using OpenAI's Whisper to get the text](example_workflows/F5TTS_whisper_workflow.json)
* [Workflow with all features](example_workflows/F5TTS-test-all.json)
* [Effortlessly Clone Your Own Voice by using ComfyUI and Almost in Real-Time! (Step-by-Step Tutorial & Workflow Included)](https://www.reddit.com/r/StableDiffusion/comments/1id8spa/effortlessly_clone_your_own_voice_by_using/)


### Other languages / custom models...

You can put the model & vocab txt files into `models/checkpoints/F5-TTS` folder if you have any more models.  Name the .txt vocab file and the .pt model file the same names.  Press "refresh" and it should appear under the "model" selection.

Example...
```
YourLanguage.txt
YourLanguage.safetensors
```

[Custom F5-TTS languages on huggingface](https://huggingface.co/models?search=f5)

I haven't tried these...
* [Finnish](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)
* [French](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)
* [German](https://huggingface.co/aihpi/F5-TTS-German)
* [Hindi](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)
* [Hungarian](https://huggingface.co/sarpba/F5-TTS-Hun)
* [Italian](https://huggingface.co/alien79/F5-TTS-italian)
* [Japanese](https://huggingface.co/Jmica/F5TTS)
* [Malaysian](https://huggingface.co/mesolitica/Malaysian-F5-TTS)
* [Norwegian](https://huggingface.co/akhbar/F5_Norwegian)
* [Polish](https://huggingface.co/Gregniuki/F5-tts_English_German_Polish/tree/main/Polish)
* [Portuguese BR](https://huggingface.co/firstpixel/F5-TTS-pt-br)
* [Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)
* [Spanish](https://huggingface.co/jpgallegoar/F5-Spanish)
* [Turkish](https://huggingface.co/marduk-ra/F5-TTS-Turkish)
* [Thai](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)
* [Vietnamese](https://huggingface.co/yukiakai/F5-TTS-Vietnamese)
* [Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu](https://huggingface.co/ShriAishu/hindiSpeech)

### Multi voices output...

Use the `F5-TTS Audio` node(not the "from input" node).

Put your sample voice files into the `input` folder like...
```
voice.wav
voice.txt
voice.deep.wav
voice.deep.txt
voice.chipmunk.wav
voice.chipmunk.txt
```

Then you can use prompts for different voices...
```
{main} Hello World this is the end
{deep} This is the narrator
{chipmunk} Please, I need more helium
```

### Multi voice input...

Put a sentence of voice 1 and a sentence from voice 2 into the input audio sample.  F5-TTS cuts the audio off at 15 seconds so don't make it too long.
[Example](https://github.com/niknah/ComfyUI-F5-TTS/issues/29)

### BigVGAN models.

To use BigVGAN, you have to add a little dot to make it work with ComfyUI...

In the file `custom_nodes/ComfyUI-F5-TTS/F5-TTS/src/third_party/BigVGAN/bigvgan.py`

Add a little dot on the line at the top that says...

`from utils import init_weights, get_padding`

so it's becomes...

`from .utils import init_weights, get_padding`


### Tips...

 * F5-TTS [cuts your voice sample off at 15 secs](https://github.com/SWivid/F5-TTS/blob/8898d05e374bcb8d3fc0b1286037e95df61f491f/src/f5_tts/infer/utils_infer.py#L315).   It may cut off in the middle of a word and not cut the text only audio.   Make sure your input samples are less than 15 secs.


### Install from git

It's best to install from ComfyUI-manager because it will update all your custom\_nodes when you click "update all".  With git, you will have to update manually.

Clone this repository into custom\_nodes and run this to install from git
```
cd custom_nodes/ComfyUI-F5-TTS
git submodule update --init --recursive
pip install -r requirements.txt
```

### No module named f5\_tts

Some versions of git doesn't handle submodules well.  Remove the `custom\_nodes/ComfyUI-F5-TTS/F5-TTS` folder and clone the F5-TTS repository...
```
cd ComfyUI/custom\_nodes/ComfyUI-F5-TTS/
rm -rf F5-TTS
rmdir /s F5-TTS
git clone https://github.com/SWivid/F5-TTS.git F5-TTS
```

### Changes

* 1.0.23: Made F5-TTSv1 the default model.
* 1.0.22: Added TDHS(Time-domain harmonic scaling) to advanced node.
* 1.0.21: Added advanced node
* 1.0.19: Added model\_type.
