from overrides import overrides
from pytorch_transformers.modeling_auto import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """
    def __init__(self, model_name: str, requires_grad: bool, requires_hidden: bool) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size
        self.requires_hidden = requires_hidden
        self.transformer_model.encoder.output_hidden_states = requires_hidden
        for param in self.transformer_model.parameters():
            param.requires_grad = requires_grad

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None,
                only_pooler = False) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        if token_type_ids is None:
            if only_pooler:
                # return pooled output of BERT   shape=[batch_size, max_length, embedding_size] = [32, 300, 768]
                return self.transformer_model(token_ids)[1]
            else:
                # return last hidden state of BERT shape=[batch_size, embedding_size] = [32, 768]
                return self.transformer_model(token_ids)[0]

        else:
            if only_pooler:
                # return pooled output of BERT   shape=[batch_size, max_length, embedding_size] = [32, 300, 768]
                return self.transformer_model(token_ids, token_type_ids=token_type_ids)[1]
            else:
                if not self.requires_hidden:
                    # return last hidden state of BERT shape=[batch_size, embedding_size] = [32, 768]
                    return self.transformer_model(token_ids, token_type_ids=token_type_ids)[0]
                else:
                    # return list of hidden states, each hidden states is of shape [batch_size, length, embedding_size]
                    return self.transformer_model(token_ids, token_type_ids=token_type_ids)[2]

