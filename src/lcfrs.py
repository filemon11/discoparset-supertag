"""TODO"""

import os.path

from typing import Literal, List, Tuple
from parsing_typing import Corpus, Sentence


def import_lcfrs(split : Literal["train"] | Literal["dev"] | Literal["test"], path : str) -> Tuple[Corpus, Corpus]:

    tokens      : Corpus = []
    supertags   : Corpus = []

    with open(os.path.join(path, f"{split}-tags"), 'r') as file:

        tokens_sent     : Sentence = []
        supertags_sent  : Sentence = []

        for line in file:
            if len(line.split()) == 0:
                tokens.append(tokens_sent)
                supertags.append(supertags_sent)

                tokens_sent = []
                supertags_sent = []
            
            else:
                components : List[str] = line.split()

                tokens_sent.append(components[0])
                supertags_sent.append(components[1])
            
        return tokens, supertags