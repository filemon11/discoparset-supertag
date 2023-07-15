from datasets import load_dataset, Dataset, DatasetDict

from typing import List, Tuple, Literal, cast



def import_chunking_data(category : Literal["test"] | Literal["train"] | Literal["dev"]) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Imports the conll2000 chunking dataset with the help of the
    huggingface dataset module. Based on Wall Street Journal corpus (WSJ)
    sections 15-18 (train, 211.727 tokens) and section 20 (test, 47.377 tokens).
    Since the dataset has no development split, the last 400 sentences 
    of the train split are provided for development. They are removed
    for "train".

    Parameters
    ----------
    category : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        The chunking values as numerical features.
    """

    dataset : DatasetDict = cast(DatasetDict, load_dataset("conll2000"))

    tokens : List[List[str]]
    labels : List[List[int]] 
    
    if category == "dev":
        tokens = dataset["train"]["tokens"][-400:]
        labels = dataset["train"]["chunk_tags"][-400:]

    elif category == "train":
        tokens = dataset["train"]["tokens"][:-400]
        labels = dataset["train"]["chunk_tags"][:-400]

    else:
        tokens = dataset["test"]["tokens"]
        labels = dataset["test"]["chunk_tags"]

    return tokens, [[label for label in sentence] for sentence in labels]