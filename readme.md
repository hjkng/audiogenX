# AudioGenX: Text-to-Audio Generation eXplainable AI

## Requirements
- python==3.9
- spacy==3.5.2
- torch>=2.1.0
- torchaudio>=2.1.0
- transformers>=4.31.0

Refer to requirements.txt for more details.

## Usage
```shell
  pip install -r requirements.txt
  
# If you want to use the model in your project, add the project path to the sys path in head of the codes
  import sys
  project_path = '' # absolute path of your project
  sys.path.append(project_path)
  
  # if you want to generate audios from scratch, run the following scripts
  python gen_audio.py # generate explanation mask and factual & counterfactual audio

  # if you want to explain generated audios
  python explain.py # explain generated_audio & evalaute factual and counterfactual audio
```

## AudioGen Model Setting
For this project, we initially used AudioGen. 
Explanation methods have not yet been implemented on MusicGen.

    AudioGen
        Defalut model   : facebook/audiogen-medium
        Duration        : 5sec
        Top_k sampling  : 250
        sample_rate     : 16000

more detail about audio generate models
* [AudioGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md): A state-of-the-art text-to-sound model.

## Metrics
    Models performance measures: We used the following objective measure to evaluate the model on a standard audio benchmark:
    - Kullback-Leibler Divergence on label distributions extracted from a pre-trained audio classifier (PaSST)


## File Structure
```
AudioGenX/

├── audiocraft/                     
        │
        └── models/                 
               ├── audiogen.py      # audigen 
               ├── lm.py            # predict sequence of audio token
               ├── mask.py          # explanation mask 
               └── explainer.py     # generate explanation mask for generated audio                     
├── data/                           # textual prompts for evaluation
├── AudioGenX_demo                  # demo 
├── gen_audio.py                    # generate audios
├── explain.py                      # train explanation mask & evaluate factual and counterfactual audios
├── example/                        # generated factual and counterfactual audios
├── readme.md                       
└── requirements.txt                
```


## License
AudioGen model and codes are from [audiocraft](https://github.com/facebookresearch/audiocraft/tree/main) by Facebook Research.  
See license information in the [model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/AUDIOGEN_MODEL_CARD.md).
