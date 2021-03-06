import pandas as pd
import torch
from transformers import BertJapaneseTokenizer
from wtfml.data_loaders.nlp.utils import clean_sentence


class BERTSimpleDataset:
    """
    Dataset for bert which can accept clearning function
    """

    def __init__(self, input_texts, target, clearning_function=clean_sentence):
        if isinstance(input_texts, pd.Series):
            input_texts = list(input_texts)
        self.input_texts = input_texts
        self.target = target
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.max_len = 144  # twitter
        self.clearning_function = clearning_function

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, item):
        input_text = str(self.input_texts[item])
        if self.clearning_function:
            input_text = self.clearning_function(input_text)

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # return_tensors="pt"
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
