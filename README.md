# DisConStackFreeTest


used dependencies incl. versions:

datasets==VERSION?

# Discoparset

This is the code for:

Integrating Supertag Features into Neural Discontinuous Constituent Parsing
Lukas Mielczarek
[[pdf]](DOWNLOAD)

## Install dependencies

Assuming you have a Python X environment.

Install disco-dop (https://github.com/andreasvc/disco-dop)
https://github.com/andreasvc/disco-dop

Install depCCG: (https://github.com/masashi-y/depccg)

    pip install pytorch torchvision cudatoolkit=9.0 -c pytorch
    # or (for cpu only)
    pip install pytorch-cpu torchvision-cpu -c pytorch

    pip intsall datasets==VERSION TODO, conllu==VERSION

## Generate datasets
    
**DPTB:**
    
    git clone https://github.com/mcoavoux/multilingual_disco_data.git
    # and follow instructions in multilingual_disco_data/readme

    You need to provide a copy of the DPTB (contact Kilian Evang).

**CCGrebank:**

Place unzipped `CCGrebank` folder into the project
directory. Path can be changed via ``-ccg`` parameter
when launching ``sfparser.py``.

**DepPTB:**

Place unzipped ``DepPTB`` folder into project directory.
Path can be changed via ``-depptb`` parameter when 
launching ``sfparser.py``.

## Reproduce result with pretrained models:

    TODO
    cd src
    # python parser.py eval <pretrained model>  <input: 1 tokenized sentence per line> <output> [--gpu <gpu_id>] [--gold gold.discbracket]
    python parser.py eval ../pretrained_models/dptb_dynamic_oracle/  ../multilingual_disco_data/data/dptb/dev.tokens ptb_dev_out --gpu 0 --gold ../multilingual_disco_data/data/dptb/dev.discbracket

    # --gpu 0  -> use gpu number 0
    # --gold gold.discbracket: evaluate output against gold corpus after parsing

    # Expected results
    ...
    precision=91.53
    recall=91.34
    fscore=91.44
    disc-precision=76.09
    disc-recall=66.37
    disc-fscore=70.9

## Reproduce results by retraining models:

    TODO