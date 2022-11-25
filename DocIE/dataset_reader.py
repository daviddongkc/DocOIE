import logging
from typing import List, Dict
import numpy as np
from overrides import overrides
from DocIE import bert_utils
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import json

logger = logging.getLogger(__name__)

@DatasetReader.register("copy_seq2seq_doc")
class CopySeq2SeqNetDocumentDatasetReader(DatasetReader):
    def __init__(self,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_tokens: int = None,
                 bert: bool = False,
                 dev_sent_path: str = None,
                 min_confidence: int = None,
                 max_confidence: int = None,
                 validation: bool = False,
                 gradients: bool = True) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self.dev_sent_path = dev_sent_path
        self._max_tokens = max_tokens
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        self._bert = bert
        self._validation = validation
        self._gradients = gradients
        if self._bert:
            self._target_token_indexers: Dict[str, TokenIndexer] = source_token_indexers
            global START_SYMBOL, END_SYMBOL
            START_SYMBOL, END_SYMBOL = bert_utils.init_globals()
        else:
            self._target_token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer(namespace=self._target_namespace)}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            data = json.load(data_file)

            for doc_num, doc in enumerate(data):
                for sent_num, sent in enumerate(data[doc]):
                    source_sequence = sent
                    line_num = sent_num
                    if self._validation:
                        if len(data[doc][sent]) == 0: continue
                        target_sequence = None
                        if not source_sequence: continue
                        instance = self.text_to_instance(source_sequence, target_sequence, line_num, doc_num, self._validation, 1)
                        if instance == None: continue
                        yield instance

                    else:
                        sent_triples = data[doc][sent]
                        for triple in sent_triples:
                            target_sequence = '<arg1> ' + triple['sub'] + ' </arg1>' + ' <rel> ' + triple['rel'] \
                                              + ' </rel>' + ' <arg2> ' + triple['obj'] + ' </arg2>'
                            confidence = triple['conf']
                            if self._min_confidence != None and confidence < self._min_confidence: continue
                            if self._max_confidence != None and confidence > self._max_confidence: continue
                            if not source_sequence: continue
                            instance = self.text_to_instance(source_sequence, target_sequence, line_num, doc_num, self._validation, confidence)
                            if instance == None: continue
                            yield instance

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text, len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None, example_id: int = None,
                         example_doc_id: int = None,
                         validation: bool = False, confidence: float = None) -> Instance:
        if self._bert:
            source_string = bert_utils.replace_strings(source_string)
            target_string = bert_utils.replace_strings(target_string)

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))

        source_field = TextField(tokenized_source, self._source_token_indexers)

        if (self._max_tokens != None and len(tokenized_source) > self._max_tokens):
            return None

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]], "example_ids": example_id,
                       "example_doc_ids": example_doc_id, "validation": validation, "confidence": confidence}
        fields_dict = {"source_tokens": source_field, "source_to_target": source_to_target_field}

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] + tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source) - 2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source) - 2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
