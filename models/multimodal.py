import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers

from models.heads import *

UNFREEZE = [
    "pooler.dense.weight",
    "pooler.dense.bias",
]

UNFREEZE_STARTSWITH = []


class MultimodalLateLateModel(nn.Module):
    """Multimodal classification model
    fusion strategy: late concatenation
    classification head: Multi-layer perceptron
    """

    def __init__(
        self,
        text_encoder_checkpoint,
        audio_encoder_checkpoint,
        n_classes=4,
        head_hidden_layers=0,
        head_hidden_size=128,
        text_encoder_dropout=0.1,
        audio_encoder_dropout=0.1,
        text_dropout=0.2,
        audio_dropout=0.2,
        activation="relu",
        freeze_encoders=False,
        initialisation="kaiming_normal",
        sequence_fusion="concatenation",
        modality_fusion="concatenation",
        **kwargs
    ):
        super().__init__()

        # load encoder configurations
        self.text_config = transformers.RobertaConfig.from_pretrained(
            text_encoder_checkpoint
        )
        self.text_config.hidden_dropout_prob = text_encoder_dropout

        self.audio_config = transformers.Wav2Vec2Config.from_pretrained(
            audio_encoder_checkpoint
        )
        self.audio_config.hidden_dropout = audio_encoder_dropout

        self.text_hidden_size = self.text_config.hidden_size
        self.audio_hidden_size = self.audio_config.hidden_size

        # load encoder models
        self.text_encoder = transformers.RobertaModel.from_pretrained(
            text_encoder_checkpoint, config=self.text_config
        )
        self.audio_encoder = transformers.Wav2Vec2Model.from_pretrained(
            audio_encoder_checkpoint, config=self.audio_config
        )

        self.audio_lstm = LSTMStack(self.audio_hidden_size, [64, 32])

        # initialise dropout layers
        self.text_dropout = nn.Dropout(p=text_dropout)
        self.audio_dropout = nn.Dropout(p=audio_dropout)

        # initialise classification head
        self.sequence_fusion = sequence_fusion
        self.modality_fusion = modality_fusion

        self.head = MLPMultilayerClassificationHead(
            self.text_hidden_size * 2 + 64 * 2,
            n_classes,
            head_hidden_size,
            head_hidden_layers,
            activation,
            initialisation,
        )

        self.freeze_encoders()

    def get_encoding(self, audio, text):
        """Method to get the encoding for a single sequence

        Args:
            audio (dict): audio data
            text (dict): text data

        Returns:
            torch.Tensor: hidden vector
        """

        # get pooled text encodings
        text_encoding_pooled = self.text_encoder(**text)[1]
        text_encoding_pooled = self.text_dropout(text_encoding_pooled)

        # get raw audio encodings
        audio_encoding = self.audio_encoder(**audio)[0]
        audio_encoding = self.audio_dropout(audio_encoding)

        # put audio encodings through lstm
        audio_encoding_pooled = self.audio_lstm(audio_encoding)

        print(audio_encoding_pooled.shape)

        # concatenate text and audio encodings
        concat_encoding = torch.cat(
            (text_encoding_pooled, audio_encoding_pooled), dim=-1
        )

        return concat_encoding

    def forward(self, audio1, text1, audio2, text2, **kwargs):
        """Model's forward method

        Args:
            audio1 (dict): audio data for sequence 1
            text1 (dict): text data for sequence 1
            audio2 (dict): audio data for sequence 2
            text2 (dict): text data for sequence 2

        Returns:
            torch.Tensor: classification logits
        """

        # get individual sequence encodings
        seq1_encoding = self.get_encoding(audio1, text1)
        seq2_encoding = self.get_encoding(audio2, text2)

        # concatenate sequence encodings into single hidden vector
        hidden_vector = torch.cat(
            (seq1_encoding, seq2_encoding),
            dim=-1,
        )

        # return classification logits
        logits = self.head(hidden_vector)

        return {"logits": logits}

    def freeze_encoders(self):
        """Method to freeze the encoders' learning"""
        # freeze text encoder
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = False

        # freeze audio encoder
        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = False

    def unfreeze_encoders(self):
        """Method to unfreeze the encoders' learning"""
        # unfreeze the text encoder
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = True

        # unfreeze the audio encoder
        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = True


class MultimodalEarlyLateModel(nn.Module):
    """Multimodal classification model
    fusion strategy: late concatenation
    classification head: Multi-layer perceptron
    """

    def __init__(
        self,
        text_encoder_checkpoint,
        audio_encoder_checkpoint,
        n_classes=4,
        head_hidden_layers=4,
        head_hidden_size=128,
        text_encoder_dropout=0.1,
        audio_encoder_dropout=0.1,
        text_dropout=0.5,
        audio_dropout=0.5,
        activation="relu",
        freeze_encoders=False,
        initialisation="kaiming_normal",
        pooling_method="mean",
        mm_fusion_method="concatenation",
    ):
        super().__init__()

        self.mm_fusion_method = mm_fusion_method
        self.pooling_method = pooling_method

        # load encoder configurations
        self.text_config = transformers.RobertaConfig.from_pretrained(
            text_encoder_checkpoint
        )
        self.text_config.hidden_dropout_prob = text_encoder_dropout

        self.audio_config = transformers.Wav2Vec2Config.from_pretrained(
            audio_encoder_checkpoint
        )
        self.audio_config.hidden_dropout = audio_encoder_dropout

        # load encoder models
        self.text_encoder = transformers.RobertaModel.from_pretrained(
            text_encoder_checkpoint, config=self.text_config
        )
        self.audio_encoder = transformers.Wav2Vec2Model.from_pretrained(
            audio_encoder_checkpoint, config=self.audio_config
        )

        # initialise dropout layers
        self.text_dropout = nn.Dropout(p=text_dropout)
        self.audio_dropout = nn.Dropout(p=audio_dropout)

        # get encoder hidden sizes
        self.text_hidden_size = self.text_config.hidden_size
        self.audio_hidden_size = self.audio_config.hidden_size

        # calculate required output size
        if mm_fusion_method == "concat":
            self.hidden_size = self.text_hidden_size + self.audio_hidden_size
        else:
            if not "ca" in mm_fusion_method:
                assert self.text_hidden_size == self.audio_hidden_size
            self.hidden_size = self.text_hidden_size

        # initialise classification head
        self.head = MLPClassificationHead(
            self.hidden_size,
            n_classes,
            head_hidden_size,
            head_hidden_layers,
            activation,
            initialisation,
        )

        # initialise cross attention module if required
        if mm_fusion_method == "ca_text":
            self.cross_attention_module = CrossModalAttention(
                self.text_hidden_size, self.audio_hidden_size, self.text_hidden_size
            )
        elif mm_fusion_method == "ca_audio":
            self.cross_attention_module = CrossModalAttention(
                self.audio_hidden_size, self.text_hidden_size, self.text_hidden_size
            )

        # allow encoders to be trained
        if freeze_encoders:
            self.freeze_encoders()

    def get_encoding(self, audio, text):
        """Method to get the encoding for a single sequence

        Args:
            audio (dict): audio data
            text (dict): text data

        Returns:
            torch.Tensor: hidden vector
        """

        # get pooled text encodings
        text_encoder_output = self.text_encoder(**text)
        text_encoding_pooled = text_encoder_output[1]
        text_encoding_pooled = self.text_dropout(text_encoding_pooled)

        # get raw audio encodings
        audio_encoding = self.audio_encoder(**audio)[0]

        # pool audio encodings using mean
        if self.pooling_method == "mean":
            audio_encoding_pooled = audio_encoding.mean(dim=1)
        elif self.pooling_method == "last":
            audio_encoding_pooled = audio_encoding[:, -1, :]
        elif self.pooling_method == "first":
            audio_encoding_pooled = audio_encoding[:, 0, :]
        elif self.pooling_method == "max":
            audio_encoding_pooled = audio_encoding.max(dim=1)
        elif self.pooling_method == "min":
            audio_encoding_pooled = audio_encoding.min(dim=1)
        else:
            raise ValueError("Invalid pooling method")

        audio_encoding_pooled = self.audio_dropout(audio_encoding_pooled)

        # perform multimodal fusion
        if self.mm_fusion_method == "concat":
            hidden_vector = torch.cat(
                (text_encoding_pooled, audio_encoding_pooled), dim=-1
            )
        elif self.mm_fusion_method == "prod":
            hidden_vector = torch.mul(text_encoding_pooled, audio_encoding_pooled)
        elif self.mm_fusion_method == "ca_text":
            ca_output = self.cross_attention_module(
                text_encoder_output[0], audio_encoding
            )

            # pool cross attention output into a hidden vector
            if self.pooling_method == "mean":
                hidden_vector = ca_output.mean(dim=1)
            elif self.pooling_method == "last":
                hidden_vector = ca_output[:, -1, :]
            elif self.pooling_method == "first":
                hidden_vector = ca_output[:, 0, :]
            elif self.pooling_method == "max":
                hidden_vector = ca_output.max(dim=1)
            elif self.pooling_method == "min":
                hidden_vector = ca_output.min(dim=1)
            else:
                raise ValueError("Invalid pooling method")

        elif self.mm_fusion_method == "ca_audio":
            ca_output = self.cross_attention_module(
                audio_encoding, text_encoder_output[0]
            )

            # pool cross attention output into a hidden vector
            if self.pooling_method == "mean":
                hidden_vector = ca_output.mean(dim=1)
            elif self.pooling_method == "last":
                hidden_vector = ca_output[:, -1, :]
            elif self.pooling_method == "first":
                hidden_vector = ca_output[:, 0, :]
            elif self.pooling_method == "max":
                hidden_vector = ca_output.max(dim=1)
            elif self.pooling_method == "min":
                hidden_vector = ca_output.min(dim=1)
            else:
                raise ValueError("Invalid pooling method")

        else:
            raise ValueError("Invalid multimodal fusion method")

        return hidden_vector

    def forward(self, audio, text, **kwargs):
        """Model's forward method

        Args:
            audio (dict): audio data
            text (dict): text data

        Returns:
            torch.Tensor: classification logits
        """

        # get individual sequence encodings
        hidden_vector = self.get_encoding(audio, text)

        # return classification logits
        logits = self.head(hidden_vector)

        return {"logits": logits}

    def freeze_encoders(self):
        """Method to freeze the encoders' learning"""
        # freeze text encoder
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = False

        # freeze audio encoder
        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = False

    def unfreeze_encoders(self):
        """Method to unfreeze the encoders' learning"""
        # unfreeze the text encoder
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = True

        # unfreeze the audio encoder
        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = True


class CrossModalAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)

    def forward(self, query_modality, kv_modality):
        # project query features
        queries = self.query_proj(query_modality)

        # project kv features
        keys = self.key_proj(kv_modality)
        values = self.value_proj(kv_modality)

        # compute attention scores
        attn_scores = torch.bmm(queries, keys.transpose(1, 2))
        attn_probs = F.softmax(attn_scores, dim=-1)

        # compute cross modal features
        cross_modal_features = torch.bmm(attn_probs, values)

        return cross_modal_features
    

class LSTMStack(nn.Module):

    def __init__(
            self,
            input_size,
            lstm_weigths,
            return_hidden=True
    ):
        super().__init__()

        self.return_hidden = return_hidden

        self.lstm = nn.ModuleList()
        for weight in lstm_weigths:
            self.lstm.append(nn.LSTM(input_size=input_size,
                                        hidden_size=weight,
                                        batch_first=True,
                                        bidirectional=True))
            input_size = weight * 2

    def forward(
            self,
            x
    ):
        hidden = None
        inputs = x
        for lstm_module in self.lstm:
            inputs, hidden = lstm_module(inputs)

        if self.return_hidden:
            # [bs, d * 2]
            last_hidden = hidden[0]
            return last_hidden.permute(1, 0, 2).reshape(x.shape[0], -1)

        # [bs, T, d]
        return inputs
