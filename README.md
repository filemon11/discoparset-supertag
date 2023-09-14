# Discoparset-Supertag

## Install dependencies

Assuming you have a Python 3.10.9 environment.

Install disco-dop (https://github.com/andreasvc/disco-dop)

Install depCCG (2.0.3.2): (https://github.com/masashi-y/depccg)
Note that the models (basic and rebank) currently need to be downloaded manually
(see https://github.com/masashi-y/depccg/issues/10). The links are in the README.md

After that run:

    pip install -r requirements.txt

## Generate datasets
    
**DPTB:**
    
    git clone https://github.com/mcoavoux/multilingual_disco_data.git

and follow instructions in multilingual_disco_data/readme

You need to provide a copy of the DPTB (contact Kilian Evang).

**CCGrebank:**

Place unzipped `CCGrebank` folder into the project
directory. Path can be changed via ``-ccg`` parameter
when launching ``sfparser.py``.

**DepPTB:**

Place unzipped ``DepPTB`` folder into project directory.
Path can be changed via ``-depptb`` parameter when 
launching ``sfparser.py``.

**LCFRS:**

Convert ``dptb7.export`` via LCFRS supertagger (https://github.com/truprecht/lcfrs-supertagger).
Extract contents of ``.corpus-cache/dptb.3914-39832-1700-2416.bin-0-1-r.Vanilla.Vanilla.tar.gz``
into a folder named ``LCFRS`` in the project directory.
Path can be changed via ``-lcfrs`` parameter when 
launching ``sfparser.py``.

**LTAG-spinal:**

Download LTAG-spinal treebank from https://www.cis.upenn.edu/~xtag/spinal/ and place
contents into a folder named ``LTAGspinal``in the project directory
Path can be changed via ``-LTAGspinal`` parameter when 
launching ``sfparser.py``.

## Download models

Due to github's file size limitations, the models are stored externally. You can download them at https://drive.google.com/drive/folders/1Gqyq9H1TihH5D5Z0WWKHeQeDse-3wSfI?usp=sharing

I recommend using ``gdown`` to download from Google Drive: https://github.com/wkentaro/gdown

## Reproduce result with pretrained models:

Each model has its own launch_experiments_\<model\>.sh script. In the script, comment out the line 
starting with ``python sfparser.py train``. Then run in terminal:

    sh launch_experiments_<model>.sh "dptb" "../pretrained_models/<model> <device>

Use -1 for CPU and 0 for GPU (assuming you have only one GPU).

## Reproduce results by retraining models:

Each model has its own launch_experiments_\<model\>.sh script. Run in terminal:

    sh launch_experiments_<model>.sh "dptb" "../pretrained_models/<model> <device>

Use -1 for CPU and 0 for GPU (assuming you have only one GPU).

## References:

This project extends the discoparset project created by

    @inproceedings{coavoux-cohen-2019-discontinuous,
    title = "Discontinuous Constituency Parsing with a Stack-Free Transition System and a Dynamic Oracle",
    author = "Coavoux, Maximin  and
      Cohen, Shay B.",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1018",
    doi = "10.18653/v1/N19-1018",
    pages = "204--217"
    }

Their code is available at https://gitlab.com/mcoavoux/discoparset
