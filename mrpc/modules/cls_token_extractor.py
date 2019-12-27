from overrides import overrides
import torch

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("cls_token")
class CLSTokenExtractor(Seq2VecEncoder):
    """
    A ``CLSTokenExtractor`` is a simple :class:`Seq2VecEncoder` which simply extract the [CLS] word embeddings.
    Parameters
    ----------
    hidden_size: ``int``
        This is the number of hidden units of BERT.
    """
    def __init__(self,
                 hidden_size: int) -> None:
        super(CLSTokenExtractor, self).__init__()
        self._hidden_size = hidden_size

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.tensor:  # pylint: disable=arguments-differ, unused-argument
        # extract CLS token
        return tokens[:, 0]

    @overrides
    def get_input_dim(self) -> int:
        return self._hidden_size

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size
