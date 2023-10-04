import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DebertaV2ForMaskedLM, DebertaV2Model, DebertaV2PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PredictionHeadTransform


@dataclass
class DebertaV3ForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    generator_loss: torch.FloatTensor | None = None
    discriminator_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class DebertaV3ForReplacedTokenDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class DebertaV3ForPreTraining(DebertaV2PreTrainedModel):
    def __init__(
        self, config: PretrainedConfig, config_generator: PretrainedConfig, loss_weight_lambda: float = 50.0
    ) -> None:
        super().__init__(config)

        self.generator = DebertaV2ForMaskedLM(config_generator)
        self.discriminator = DebertaV3ForReplacedTokenDetection(config)
        self.loss_weight_lambda = loss_weight_lambda

        self.register_discriminator_forward_pre_hook()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        self.init_discriminator_embeddings()

    def init_discriminator_embeddings(self):
        """initialize discriminator's embedding for Gradient-Disentangled Embedding Sharing"""

        def set_embeddings_weight_delta(embeddings: nn.Embedding):
            embeddings.register_parameter("weight_delta", nn.Parameter(torch.zeros_like(embeddings.weight)))
            delattr(embeddings, "weight")

        set_embeddings_weight_delta(self.discriminator.deberta.embeddings.word_embeddings)
        if self.config.position_biased_input:
            set_embeddings_weight_delta(self.discriminator.deberta.embeddings.position_embeddings)
        if self.config.type_vocab_size > 0:
            set_embeddings_weight_delta(self.discriminator.deberta.embeddings.token_type_embeddings)

    def register_discriminator_forward_pre_hook(self):
        """register forward pre hook to set discriminator's embedding with for Gradient-Disentangled Embedding Sharing"""

        def set_embeddings_weight_added_delta_as_buffer(
            discriminator_embeddings: nn.Embedding, generator_embeddings_weight: nn.Parameter
        ):
            if hasattr(discriminator_embeddings, "weight"):
                delattr(discriminator_embeddings, "weight")
            discriminator_embeddings.register_buffer(
                "weight", generator_embeddings_weight.detach() + discriminator_embeddings.weight_delta
            )

        def forward_pre_hook(module, *inputs):
            set_embeddings_weight_added_delta_as_buffer(
                self.discriminator.deberta.embeddings.word_embeddings,
                self.generator.deberta.embeddings.word_embeddings.weight[:-1],
            )
            if self.config.position_biased_input:
                set_embeddings_weight_added_delta_as_buffer(
                    self.discriminator.deberta.embeddings.position_embeddings,
                    self.generator.deberta.embeddings.position_embeddings.weight,
                )
            if self.config.type_vocab_size > 0:
                set_embeddings_weight_added_delta_as_buffer(
                    self.discriminator.deberta.embeddings.token_type_embeddings,
                    self.generator.deberta.embeddings.token_type_embeddings.weight,
                )

        self.discriminator.register_forward_pre_hook(forward_pre_hook)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | DebertaV3ForPreTrainingOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_outputs = self.generator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        with torch.no_grad():
            masked_positions = labels != -100
            generator_logits = generator_outputs.logits if return_dict else generator_outputs[1]
            generator_predictions = torch.argmax(generator_logits[masked_positions], dim=-1)

            discriminator_input_ids = input_ids.clone()
            discriminator_input_ids[masked_positions] = generator_predictions

            discriminator_labels = None
            if labels is not None:
                discriminator_labels = torch.zeros_like(
                    discriminator_input_ids, dtype=torch.long, device=discriminator_input_ids.device
                )
                discriminator_labels[masked_positions] = (labels[masked_positions] != generator_predictions).long()

        discriminator_outputs = self.discriminator(
            discriminator_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=discriminator_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            if return_dict:
                loss = generator_outputs.loss + self.loss_weight_lambda * discriminator_outputs.loss
            else:
                loss = generator_outputs[0] + self.loss_weight_lambda * discriminator_outputs[0]

        if not return_dict:
            return (
                (loss, generator_outputs[0], *discriminator_outputs) if labels is not None else discriminator_outputs
            )

        return DebertaV3ForPreTrainingOutput(
            loss=loss,
            generator_loss=generator_outputs.loss,
            discriminator_loss=discriminator_outputs.loss,
            logits=discriminator_outputs.logits,
            hidden_states=discriminator_outputs.hidden_states,
            attentions=discriminator_outputs.attentions,
        )

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs):
        self.generator.save_pretrained(os.path.join(save_directory, "generator"), **kwargs)
        self.save_pretrained_discriminator(save_directory, **kwargs)

    def save_pretrained_discriminator(self, save_directory: str | os.PathLike, **kwargs):
        """save discriminator's weights with for Gradient-Disentangled Embedding Sharing"""

        def set_embeddings_weight_added_delta_as_parameter(
            discriminator_embeddings: nn.Embedding, generator_embeddings_weight: nn.Parameter
        ):
            if hasattr(discriminator_embeddings, "weight"):
                delattr(discriminator_embeddings, "weight")
            discriminator_embeddings.register_parameter(
                "weight",
                nn.Parameter(generator_embeddings_weight.detach() + discriminator_embeddings.weight_delta.detach()),
            )

        set_embeddings_weight_added_delta_as_parameter(
            self.discriminator.deberta.embeddings.word_embeddings,
            self.generator.deberta.embeddings.word_embeddings.weight[:-1],
        )
        if self.config.position_biased_input:
            set_embeddings_weight_added_delta_as_parameter(
                self.discriminator.deberta.embeddings.position_embeddings,
                self.generator.deberta.embeddings.position_embeddings.weight,
            )

        if self.config.type_vocab_size > 0:
            set_embeddings_weight_added_delta_as_parameter(
                self.discriminator.deberta.embeddings.token_type_embeddings,
                self.generator.deberta.embeddings.token_type_embeddings.weight,
            )

        self.discriminator.save_pretrained(save_directory, **kwargs)

        delattr(self.discriminator.deberta.embeddings.word_embeddings, "weight")
        if self.config.position_biased_input:
            delattr(self.discriminator.deberta.embeddings.position_embeddings, "weight")
        if self.config.type_vocab_size > 0:
            delattr(self.discriminator.deberta.embeddings.token_type_embeddings, "weight")


class DebertaV3ForReplacedTokenDetection(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.discriminator_predictions = DebertaV3RTDHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | DebertaV3ForPreTrainingOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.discriminator_predictions(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss, *output)) if loss is not None else output

        return DebertaV3ForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV3RTDHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        logits = self.classifier(hidden_states).squeeze(-1)
        return logits
