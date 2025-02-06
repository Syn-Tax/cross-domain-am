import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers

class TextOnlyModel(nn.Module):
    """text only classification model
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
        text_encoder_dropout=0.2,
        audio_encoder_dropout=0.2,
        text_dropout=0.5,
        audio_dropout=0.5,
        activation="relu",
        freeze_encoders=False,
        initialisation="kaiming_normal",
    ):
        super().__init__()

        # load encoder configurations
        self.text_config = transformers.RobertaConfig.from_pretrained(
            text_encoder_checkpoint
        )
        self.text_config.hidden_dropout_prob = text_encoder_dropout

        # load encoder models
        self.text_encoder = transformers.RobertaModel.from_pretrained(
            text_encoder_checkpoint, config=self.text_config
        )

        # initialise dropout layers
        self.text_dropout = nn.Dropout(p=text_dropout)

        # initialise classification head
        self.text_hidden_size = self.text_config.hidden_size
        self.head = MLPClassificationHead(
            self.text_hidden_size,
            n_classes,
            head_hidden_size,
            head_hidden_layers,
            activation,
            initialisation,
        )

        # allow encoders to be trained
        if freeze_encoders:
            self.freeze_encoders()
        else:
            self.unfreeze_encoders()

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

        return text_encoding_pooled

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
        # seq1_encoding = self.get_encoding(audio1, text1)
        # seq2_encoding = self.get_encoding(audio2, text2)

        # concatenate sequence encodings into single hidden vector
        # hidden_vector = torch.cat(
        #     (seq1_encoding, seq2_encoding),
        #     dim=-1,
        # )

        hidden_vector = self.get_encoding(audio1, text1)

        # return classification logits
        logits = self.head(hidden_vector)

        return {'logits': logits}

    def freeze_encoders(self):
        """Method to freeze the encoders' learning"""
        # freeze text encoder
        for name, param in self.text_encoder.named_parameters():
            print(name)
            if name in UNFREEZE:
                param.requires_grad = True
            elif any([name.startswith(n) for n in UNFREEZE_STARTSWITH]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoders(self):
        """Method to unfreeze the encoders' learning"""
        # unfreeze the text encoder
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = True