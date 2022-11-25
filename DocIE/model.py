import logging
from typing import Dict, Tuple, List, Any, Union
import numpy as np
from overrides import overrides
import itertools
import torch
from torch.nn.modules.linear import Linear
from torch.nn import LSTM
import allennlp
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.div_beam_search import DivBeamSearch
from allennlp.nn.cov_beam_search import CoverageBeamSearch
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
import json
from DocIE import bert_utils

logger = logging.getLogger(__name__)

class KeyDict(dict):
    def __missing__(self, key):
        return key

@Model.register("copy_seq2seq_doc")
class CopyNetSeq2SeqDoc(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 bottom_attention: Seq2SeqEncoder or TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 top_encoder: bool = True,
                 document_tokenizer: Tokenizer = None,
                 document_token_indexers: Dict[str, TokenIndexer] = None,
                 train_document_path: str = '',
                 validation_document_path: str = '',
                 max_length: int = 320,
                 context_window: int = 5,
                 max_sent_length: int = 30,
                 target_embedding_dim: int = 30,
                 decoder_layers: int = 3,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 lambda_diversity: int = 5,
                 beam_search_type: str = "beam_search",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 bert: bool = False,
                 max_extractions: int = -1,
                 decoder_config: str = '',
                 decoder_type: str = 'lstm',
                 teacher_forcing: bool = True) -> None:
        super().__init__(vocab)
        self._decoder_type = decoder_type
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._bert = bert
        global START_SYMBOL, END_SYMBOL
        self._document_tokenizer = document_tokenizer  # @Tokenizer.register("pretrained_transformer")
        self._document_token_indexers = document_token_indexers  # @TokenIndexer.register("pretrained_transformer")
        self._train_document_path = train_document_path
        self._validation_document_path = validation_document_path
        if self._bert:
            START_SYMBOL, END_SYMBOL = bert_utils.init_globals()
            self._target_vocab_size = 28996
            self.token_mapping = bert_utils.mapping
        else:
            if self.vocab.get_token_index(copy_token, self._target_namespace) == 1:
                # +1 is for the copy token, which is added later on in initialize()
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace) + 1
            else:
                # copy_token already in dictionary, as an existing vocabulary is being loaded
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)
            self.token_mapping = KeyDict()
        # Encoding modules.
        self._source_embedder = source_embedder  # "type": "pretrained_transformer", "model_name": "bert-base-cased"
        self._top_encoder = top_encoder
        if self._top_encoder:
            self._bottom_attention = bottom_attention
        self._encoder = encoder
        self._attention = attention
        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps
        self._target_embedding_dim = target_embedding_dim
        self._decoder_layers = decoder_layers
        self._copy_token = copy_token
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric
        self._lambda_diversity = lambda_diversity
        self._beam_search_type = beam_search_type
        self._initializer = initializer
        self._max_extractions = max_extractions
        self._decoder_config = decoder_config
        self._decoder_type = decoder_type
        self._teacher_forcing = teacher_forcing
        self._max_length = max_length
        self._max_sent_length = max_sent_length
        self._context_window = context_window
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim
        self._target_embedder = Embedding(self._target_vocab_size, self._target_embedding_dim)
        self._input_projection_layer = Linear(self._target_embedding_dim + self.encoder_output_dim * 2, self.decoder_input_dim)

        if self._decoder_type == 'lstm':
            self._decoder_cell = LSTM(self.decoder_input_dim, self.decoder_output_dim, num_layers=self._decoder_layers, batch_first=True)
        elif self._decoder_type == 'transformer':
            decoder_layer = torch.nn.TransformerDecoderLayer(d_model=256, nhead=4)
            self._decoder_cell = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)

        self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)
        self.train_document = self._get_document_index(self._train_document_path)
        self.validation_document = self._get_document_index(self._validation_document_path)

        self._initializer(self)
        self._initialized = False

    def _get_document_index(self, doc_path):
        with open(doc_path, "r") as data_file:
            logger.info("Creating document from lines in file at: %s", doc_path)
            data = json.load(data_file)
            train_document = []
            for doc_num, doc in enumerate(data):
                sent_list = []
                for sent_num, sent in enumerate(data[doc]):
                    sent_field = self._text_to_instance(sent)
                    if sent_field is not None:
                        sent_list.append([sent, sent_field])

                train_document.append(sent_list)
            return train_document

    def _text_to_instance(self, sent):
        max_sent_length = self._max_sent_length
        if self._bert:
            sent_string = bert_utils.replace_strings(sent)
            tokenized_sent = self._document_tokenizer.tokenize(sent_string)
            if len(tokenized_sent) > max_sent_length - 1:
                tokenized_sent = tokenized_sent[:max_sent_length - 1]
            tokenized_sent.append(Token(END_SYMBOL))
            sent_field = TextField(tokenized_sent, self._document_token_indexers)
            sent_field.index(self.vocab)
            return sent_field
        return None







    def initialize(self):
        # Initilaization which require the vocabulary to be built
        # If module requires to be placed on GPU, should be created in __init__ function itself
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        if self._bert:
            # source and target vocabulary are same - no need to use instance dictionary to get the word (it is unk there as well)
            # therefore map oov_token to a random token - not supposed to be used at all in case of bert
            self.vocab._oov_token = '[unused99]'
            self.vocab._padding_token = '[PAD]'
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
        self._copy_index = self.vocab.add_token_to_namespace(self.token_mapping[self._copy_token], self._target_namespace)
        self._eoe_index = self.vocab.get_token_index(self.token_mapping['EOE'], self._target_namespace)
        self.start_arg1 = self.vocab.get_token_index(self.token_mapping['<arg1>'], self._target_namespace)
        self.end_arg1 = self.vocab.get_token_index(self.token_mapping['</arg1>'], self._target_namespace)
        self.start_arg2 = self.vocab.get_token_index(self.token_mapping['<arg2>'], self._target_namespace)
        self.end_arg2 = self.vocab.get_token_index(self.token_mapping['</arg2>'], self._target_namespace)
        self.start_rel = self.vocab.get_token_index(self.token_mapping['<rel>'], self._target_namespace)
        self.end_rel = self.vocab.get_token_index(self.token_mapping['</rel>'], self._target_namespace)

        # At prediction time, we'll use a beam search to find the best target sequence.
        if self._beam_search_type == 'beam_search':
            self._beam_search = BeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size)
        elif self._beam_search_type == 'div_beam_search':
            self._beam_search = DivBeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size, lambda_diversity=self._lambda_diversity,
                                              ignore_indices=[self.start_arg1, self.start_arg2, self.start_rel, self.end_arg1, self.end_arg2, self.end_rel])
        elif self._beam_search_type == 'cov_beam_search':
            self._beam_search = CoverageBeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size)

    @staticmethod
    def padding_sequence(source_tokens, batch_sent_list, max_length=320):
        source_tokens = source_tokens['tokens']
        get_cuda_device = source_tokens.get_device()
        source_list = source_tokens.tolist()
        # remove if there is any padding added to source_tokens
        source_list_no_padding = []
        for x in source_list:
            source_list_no_padding.append([i for i in x if i != 0])
        segment0, segment1, token_id, type_id = [], [], [], []
        # get segment 0 list for sentence
        for x in source_list_no_padding:
            segment0.append([0 for i in x])
        # get segment 1 list for sentence context
        for x in batch_sent_list:
            segment1.append([1 for i in x])
        # concat sentence and sentence context; and add padding to the max length allowed
        for x in zip(source_list_no_padding, batch_sent_list):
            row = list(itertools.chain(*x))
            while len(row) < max_length:
                row += [0]
            if len(row) > max_length:
                SEP_token = row.pop()
                row = row[:max_length - 1]
                row += [SEP_token]
            token_id.append(row)
        # concat segment 0 and 1; and add padding to the max length allowed
        for x in zip(segment0, segment1):
            row = list(itertools.chain(*x))
            while len(row) < max_length:
                row += [1]
            if len(row) > max_length:
                row = row[:max_length]
            type_id.append(row)
        # convert the token id and token type id to torch tensor
        token_id_tensor = torch.LongTensor(token_id).to(get_cuda_device)
        type_id_tensor = torch.LongTensor(type_id).to(get_cuda_device)
        return {'tokens': token_id_tensor}, type_id_tensor



    def get_context_tokens(self, metadata):
        window = self._context_window
        batch_sent_list = []
        for item in metadata:
            if item['validation']:
                doc = self.validation_document[item['example_doc_ids']]
            else:
                doc = self.train_document[item['example_doc_ids']]
            sent_num = int(item['example_ids'])

            # select contiguous sentence before and after
            context_start = sent_num - window if (sent_num - window) > 0 else 0
            context_end = sent_num + window if (sent_num + window) < len(doc) else len(doc)
            document_sents = doc[context_start:sent_num] + doc[sent_num + 1:context_end]

            docu_sents_token = []
            for _, docu_sent in document_sents:
                sent_tokens = docu_sent.__dict__['_indexed_tokens']['tokens']
                docu_sents_token.extend(sent_tokens)

            batch_sent_list.append(docu_sents_token)

        return batch_sent_list

    def forward_single(self, source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids):

        # get related context information from input text
        batch_sent_list = self.get_context_tokens(metadata)
        # combine source with context; and add padding and prepare segment id for bert embedder
        source_context_tokens, type_id_tensor = self.padding_sequence(source_tokens, batch_sent_list, max_length=self._max_length)

        state = self._encode(source_tokens, source_context_tokens, type_id=type_id_tensor)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, target_token_ids, state)
        else:
            output_dict = {}
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

        output_dict["metadata"] = metadata

        return output_dict

    global gm

    @overrides
    def forward(self, source_tokens: Dict[str, torch.LongTensor], source_token_ids: torch.Tensor, source_to_target: torch.Tensor, metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None, target_token_ids: torch.Tensor = None, optimizer=None) -> Dict[str, torch.Tensor]:

        if not self._initialized:
            self.initialize()
            self._initialized = True

        if self.training and not self._decoder_type == 'transformer':  # useful when ngpus > 1 # produces undesirable warnings during testing - but can't avoid it
            self._decoder_cell.flatten_parameters()

        output_dict = self.forward_single(source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids)

        if metadata[0]['validation']:
            predicted_tokens = self._get_predicted_tokens(output_dict["predictions"], metadata, n_best=5)
            predicted_confidences = output_dict['predicted_log_probs']
            self._token_based_metric(predicted_tokens, predicted_confidences, [x["example_doc_ids"] for x in metadata], [x["example_ids"] for x in metadata])

        return output_dict

    def _gather_extended_gold_tokens(self, target_tokens: torch.Tensor, source_token_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.LongTensor:
        batch_size, target_sequence_length = target_tokens.size()
        trimmed_source_length = source_token_ids.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = (target_tokens == self._oov_index)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_source_token_ids = source_token_ids .unsqueeze(1).expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_target_token_ids = target_token_ids.unsqueeze(-1).expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        matches = (expanded_source_token_ids == expanded_target_token_ids)
        # shape: (batch_size, target_sequence_length)
        copied = (matches.sum(-1) > 0)
        # shape: (batch_size, target_sequence_length)
        mask = (oov & copied).long()
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) * matches).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_tokens = target_tokens * (1 - mask) + (first_match.long() + self._target_vocab_size) * mask
        return new_target_tokens

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # shape: (batch_size, encoder_output_dim)
        if not isinstance(self._encoder,allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper.PytorchSeq2SeqWrapper):  ## Assuming only LSTM as the Pytorch wrapper we will us
            ## Taking the intial hidden state corresponding to the CLS token
            final_encoder_output = state["encoder_outputs"][:, 0, :]
        else:
            # Initialize the decoder hidden state with the final output of the encoder, and the decoder context with zeros.
            final_encoder_output = util.get_final_encoder_states(state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional())

        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)
        state["decoder_hidden_all"] = final_encoder_output.unsqueeze(0).repeat(self._decoder_layers, 1, 1)
        state["decoder_context_all"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_layers, self.decoder_output_dim)
        state["decoder_hidden_all"] = state["decoder_hidden_all"].transpose(0, 1).contiguous().view(-1,self._decoder_layers * self.decoder_output_dim)
        state["decoder_context_all"] = state["decoder_context_all"].transpose(0, 1).contiguous().view(-1,self._decoder_layers * self.decoder_output_dim)

        return state

    def _encode(self, source_tokens: Dict[str, torch.Tensor], source_context_tokens: Dict[str, torch.Tensor],
                type_id: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        embedded_input = self._source_embedder(source_context_tokens, token_type_ids=type_id, only_pooler=False)
        embedded_input = embedded_input[:, :source_mask.shape[1], :]

        if self._top_encoder:
            # stacked layers of self-attention of sentence
            embedded_input = self._bottom_attention(embedded_input, source_mask)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch_size, num_inp_words, _ = state['encoder_outputs'].shape
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._decoder_type == 'lstm':
            # shape: (group_size, max_input_sequence_length)
            attentive_weights = self._attention(state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)

            # shape: (group_size, encoder_output_dim)
            attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
            # shape: (group_size, encoder_output_dim)
            selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)
            # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
            decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
            # shape: (group_size, decoder_input_dim)
            projected_decoder_input = self._input_projection_layer(decoder_input)

            # (batch_size, decoder_layers * output_dim) --> (decoder_layers, batch_size, output_dim)
            state["decoder_hidden_all"] = state["decoder_hidden_all"].view(-1, self._decoder_layers, self.decoder_output_dim).transpose(0, 1).contiguous()
            state["decoder_context_all"] = state["decoder_context_all"].view(-1, self._decoder_layers, self.decoder_output_dim).transpose(0, 1).contiguous()

            _, (state["decoder_hidden_all"], state["decoder_context_all"]) = self._decoder_cell(projected_decoder_input.unsqueeze(1), (state["decoder_hidden_all"], state["decoder_context_all"]))

            state["decoder_hidden"], state["decoder_context"] = state["decoder_hidden_all"][-1], state["decoder_context_all"][-1]
            # (decoder_layers, batch_size, output_dim) --> (batch_size, decoder_layers * output_dim)
            state["decoder_hidden_all"] = state["decoder_hidden_all"].transpose(0, 1).contiguous().view(-1, self._decoder_layers * self.decoder_output_dim)
            state["decoder_context_all"] = state["decoder_context_all"].transpose(0, 1).contiguous().view(-1, self._decoder_layers * self.decoder_output_dim)

        elif self._decoder_type == 'transformer':
            if "inputs_so_far" not in state:
                state["inputs_so_far"] = embedded_input.unsqueeze(1)
            else:
                state["inputs_so_far"] = torch.cat((state["inputs_so_far"], embedded_input.unsqueeze(1)), dim=1)
            # transformer accepts sequence-first
            outputs = self._decoder_cell(state["inputs_so_far"].transpose(0, 1), state["encoder_outputs"].transpose(0, 1))
            # consider only the hidden state corresponding to the last token
            state["decoder_hidden"] = outputs.transpose(0, 1)[:, -1, :]
        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length - 2, encoder_output_dim)
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = self._output_copying_layer(trimmed_encoder_outputs)
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)
        # shape: (batch_size, max_input_sequence_length - 2)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        return copy_scores

    def _get_ll_contrib(self,
                        generation_scores: torch.Tensor,
                        generation_scores_mask: torch.Tensor,
                        copy_scores: torch.Tensor,
                        target_tokens: torch.Tensor,
                        target_to_source: torch.Tensor,
                        copy_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores that just represent padding.
        # We apply the mask to the concatenation of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores_mask, copy_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create a selective read state.
        # shape: (batch_size, trimmed_source_length)
        copy_log_probs = log_probs[:, target_size:] + (target_to_source.float() + 1e-45).log()
        # Since `log_probs[:, target_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        selective_weights = util.masked_softmax(log_probs[:, target_size:], target_to_source)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = ((target_tokens != self._oov_index) | (target_to_source.sum(-1) == 0)).float()
        log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + trimmed_source_length)
        combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = util.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def _forward_loss(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"].size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        # shape: (batch_size, trimmed_source_length)
        copy_mask = source_mask[:, 1:-1].float()
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, trimmed_source_length)
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, trimmed_source_length)
        target_to_source = state["source_token_ids"].new_zeros(copy_mask.size())

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = state["decoder_hidden"].new_full((batch_size, self._target_vocab_size),
                                                                  fill_value=1.0)

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = ((input_choices == self._oov_index) &
                          (target_to_source.sum(-1) > 0)).long()
                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, trimmed_source_length)
                target_to_source = state["source_token_ids"] == target_token_ids[:, timestep + 1].unsqueeze(-1)
            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            # Get copy scores for each token in the source sentence, excluding the start and end tokens.
            # shape: (batch_size, trimmed_source_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"][:, timestep + 1]
            step_log_likelihood, selective_weights = self._get_ll_contrib(generation_scores,generation_scores_mask,copy_scores,step_target_tokens,target_to_source,copy_mask)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:].float()
        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss, "probs": (log_likelihood / target_mask.sum(dim=1)).tolist()}

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, source_length = state["source_mask"].size()
        trimmed_source_length = source_length - 2
        # Initialize the copy scores to zero.
        state["copy_log_probs"] = (state["decoder_hidden"].new_zeros((batch_size, trimmed_source_length)) + 1e-45).log()
        # shape: (batch_size,)
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        if (self._beam_search_type == 'cov_beam_search'):
            all_top_k_predictions, log_probabilities, all_top_k_word_log_probabilities = self._beam_search.search(start_predictions, state, self.take_search_step)
            output_dict = {"predicted_log_probs": all_top_k_word_log_probabilities,"predictions": all_top_k_predictions, 'token_scores': all_top_k_word_log_probabilities}
        else:
            all_top_k_predictions, log_probabilities = self._beam_search.search(start_predictions, state, self.take_search_step)
            target_mask = all_top_k_predictions != self._end_index
            log_probabilities = log_probabilities / (target_mask.sum(dim=2).float() + 1)  # +1 for predicting the last SEP token
            output_dict = {"predicted_log_probs": log_probabilities, "predictions": all_top_k_predictions}
        return output_dict

    def _get_input_and_selective_weights(self, last_predictions: torch.LongTensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.LongTensor, torch.Tensor]:

        group_size, trimmed_source_length = state["source_to_target"].size()

        # This is a mask indicating which last predictions were copied from the the source AND not in the target vocabulary (OOV).
        # (group_size,)
        only_copied_mask = (last_predictions >= self._target_vocab_size).long()

        # If the last prediction was in the target vocab or OOV but not copied, we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = only_copied_mask.new_full((group_size,), fill_value=self._copy_index)
        input_choices = last_predictions * (1 - only_copied_mask) + copy_input_choices * only_copied_mask

        # In order to get the `selective_weights`, we need to find out which predictions
        # were copied or copied AND generated, which is the case when a prediction appears
        # in both the source sentence and the target vocab. But whenever a prediction
        # is in the target vocab (even if it also appeared in the source sentence),
        # its index will be the corresponding target vocab index, not its index in
        # the source sentence offset by the target vocab size. So we first
        # use `state["source_to_target"]` to get an indicator of every source token
        # that matches the predicted target token.
        # shape: (group_size, trimmed_source_length)

        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(group_size, trimmed_source_length)
        # shape: (group_size, trimmed_source_length)
        source_copied_and_generated = (state["source_to_target"] == expanded_last_predictions).long()

        # In order to get indicators for copied source tokens that are OOV with respect
        # to the target vocab, we'll make use of `state["source_token_ids"]`.
        # First we adjust predictions relative to the start of the source tokens.
        # This makes sense because predictions for copied tokens are given by the index of the copied
        # token in the source sentence, offset by the size of the target vocabulary.
        # shape: (group_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size
        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * only_copied_mask
        # shape: (group_size, trimmed_source_length)
        source_token_ids = state["source_token_ids"]
        # shape: (group_size, trimmed_source_length)
        adjusted_prediction_ids = source_token_ids.gather(-1, adjusted_predictions.unsqueeze(-1))
        # This mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (group_size, trimmed_source_length)
        source_only_copied = (source_token_ids == adjusted_prediction_ids).long()
        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        source_only_copied = source_only_copied * only_copied_mask.unsqueeze(-1)

        # shape: (group_size, trimmed_source_length)
        mask = source_only_copied | source_copied_and_generated
        # shape: (group_size, trimmed_source_length)
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)

        return input_choices, selective_weights

    def _gather_final_log_probs(self, generation_log_probs: torch.Tensor, copy_log_probs: torch.Tensor,
                                state: Dict[str, torch.Tensor]) -> torch.Tensor:

        _, trimmed_source_length = state["source_to_target"].size()
        source_token_ids = state["source_token_ids"]

        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, trimmed_source_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = state["source_to_target"][:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score to the OOV token.
            copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
            copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
            combined_scores = util.logsumexp(torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
            generation_log_probs = generation_log_probs.scatter(-1,source_to_target_slice.unsqueeze(-1),combined_scores.unsqueeze(-1))
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurences = (source_token_ids[:, (i + 1):] == source_token_ids[:, i].unsqueeze(-1)).float()  # pylint: disable=line-too-long
                # shape: (group_size, trimmed_source_length - i)
                future_copy_log_probs = copy_log_probs[:, (i + 1):] + (source_future_occurences + 1e-45).log()
                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat((copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1)
                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[:, i].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        _, trimmed_source_length = state["source_to_target"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied, there will be all zeros.
        # shape: (group_size,), (group_size, trimmed_source_length)
        input_choices, selective_weights = self._get_input_and_selective_weights(last_predictions, state)
        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source sentence, excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"][:, 1:-1].float()
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_log_probs, copy_log_probs = log_probs.split(
            [self._target_vocab_size, trimmed_source_length], dim=-1)
        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_log_probs"] = copy_log_probs
        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + trimmed_source_length)
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state

    def _get_predicted_tokens(self,
                              predicted_indices: Union[torch.Tensor, np.ndarray],
                              batch_metadata: List[Any],
                              n_best: int = None) -> List[Union[List[List[str]], List[str]]]:

        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                if self._end_index in indices:
                    indices = indices[indices != self._end_index]
                    indices = list(indices)
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"], output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))
        return all_metrics
