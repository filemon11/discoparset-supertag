
# Discoparset

This is the code for:

Discontinuous Constituency Parsing with a Stack-free Transition System and a Dynamic Oracle  
Maximin Coavoux, Shay B. Cohen  
NAACL 2019.  
[[pdf]](https://www.aclweb.org/anthology/N19-1018.pdf) [[bib]](https://www.aclweb.org/anthology/N19-1018.bib) [[abs/preprint]](https://arxiv.org/abs/1904.00615) [[hal]](https://hal.archives-ouvertes.fr/hal-02150076)

## Install dependencies

Assuming you have a conda distribution of Python:

    conda create --name discoparset python=3.6
    conda activate discoparset
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    # or (for cpu only)
    conda install pytorch-cpu torchvision-cpu -c pytorch
    pip install disco-dop

## Generate datasets

    git clone https://github.com/mcoavoux/multilingual_disco_data.git
    # and follow instructions in multilingual_disco_data/readme

## Reproduce results in paper with pretrained models:

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



