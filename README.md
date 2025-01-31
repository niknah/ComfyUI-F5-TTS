
ComfyUI node to make text to speech audio with your own voice.

Using F5-TTS https://github.com/SWivid/F5-TTS


### Instructions

* Put in ComfyUI's "input" folder a .wav file of an audio of the voice you'd like to use, remove any background music, noise.
* And a .txt file of the same name with what was said.
* Press refresh to see it in the node

You can use the examples here...
* [Examples voices](examples/)
* [Simple workflow](examples/simple_ComfyUI_F5TTS_workflow.json)
* [Workflow with input audio only, using OpenAI's Whisper to get the text](examples/F5TTS_whisper_workflow.json)
* [Workflow with all features](examples/F5TTS-test-all.json)


### Other languages / custom models...

You can put the model & vocab txt files into "models/checkpoints/F5-TTS" folder if you have any more models.  Name the .txt vocab file and the .pt model file the same names.  Press "refresh" and it should appear under the "model" selection.

[Custom F5-TTS languages on huggingface](https://huggingface.co/models?search=f5)

I haven't tried these...
[Finnish](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)
[French](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)
[German](https://huggingface.co/aihpi/F5-TTS-German)
[Greek](https://huggingface.co/PetrosStav/F5-TTS-Greek)
[Hindi](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)
[Hungarian](https://huggingface.co/sarpba/F5-TTS-Hun)
[Italian](https://huggingface.co/alien79/F5-TTS-italian)
[Japanese](https://huggingface.co/Jmica/F5TTS)
[Malaysian](https://huggingface.co/mesolitica/Malaysian-F5-TTS)
[Norwegian](https://huggingface.co/akhbar/F5_Norwegian)
[Polish](https://huggingface.co/Gregniuki/F5-tts_English_German_Polish/tree/main/Polish)
[Portuguese BR](https://huggingface.co/firstpixel/F5-TTS-pt-br)
[Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)
[Spanish](https://huggingface.co/jpgallegoar/F5-Spanish)
[Turkish](https://huggingface.co/marduk-ra/F5-TTS-Turkish)
[Vietnamese](https://huggingface.co/yukiakai/F5-TTS-Vietnamese)

### Multi voices...

Put your sample voice files into the "input" folder like...
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


### Tips...

 * F5-TTS [cuts your voice sample off at 15 secs](https://github.com/SWivid/F5-TTS/blob/8898d05e374bcb8d3fc0b1286037e95df61f491f/src/f5_tts/infer/utils_infer.py#L315).


### Install from git

It's best to install from ComfyUI-manager because it will update all your custom\_nodes when you click "update all".  With git, you will have to update manually.

Clone this repository into custom\_nodes and run this to install from git
```
cd custom_nodes/ComfyUI-F5-TTS
git submodule update --init --recursive
pip install -r requirements.txt
```


