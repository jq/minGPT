from typing import Optional

import transformers
from torch import nn
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
print(classifier("I love this!"))


class EModel(nn.Module):
    def __init__(self):
        super(transformers.DistilBertPreTrainedModel, self).__init__()
        self.model = transformers.DistilBertPreTrainedModel.\
            from_pretrained("j-hartmann/emotion-english-distilroberta-base")

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None
    ):

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
