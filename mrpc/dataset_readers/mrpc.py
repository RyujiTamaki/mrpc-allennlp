from typing import Dict
import logging
import csv

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("mrpc")
class MRPCDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="bert-base-uncased")
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="bert-base-uncased")}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter="\t")
            for row in tsv_in:
                if len(row) == 5:
                    yield self.text_to_instance(text_1=row[3], text_2=row[4], label=row[0])

    @overrides
    def text_to_instance(
        self,
        text_1: str,
        text_2: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize_sentence_pair(text_1, text_2)
        fields["tokens"] = TextField(tokenized_text, self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)
