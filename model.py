import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import transformers


class ClassificationHead(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, n_classes)

    def forward(self, x):
        out = F.softmax(self.fc(x), dim=1)
        return out


class ConcatModel(nn.Module):
    def __init__(
        self,
        text_encoder_checkpoint,
        audio_encoder_checkpoint,
        text_hidden_size,
        audio_hidden_size,
        n_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.text_encoder = transformers.AutoModel.from_pretrained(
            text_encoder_checkpoint
        )
        self.audio_encoder = transformers.AutoModel.from_pretrained(
            audio_encoder_checkpoint
        )
        # self.text_layer = nn.TransformerEncoderLayer(128, 8, batch_first=True)
        # self.text_encoder = nn.TransformerEncoder(self.text_layer, 4)

        # self.audio_layer = nn.TransformerEncoderLayer(499, 8, batch_first=True)
        # self.audio_encoder = nn.TransformerEncoder(self.audio_layer, 4)

        self.text_hidden_size = text_hidden_size
        self.audio_hidden_size = audio_hidden_size

        self.dropout = nn.Dropout(p=dropout)

        self.head = ClassificationHead(
            text_hidden_size * 2 + audio_hidden_size * 2, n_classes
        )

        self.freeze_encoders()

    def forward(self, audio1, text1, audio2, text2, **kwargs):
        batch_size = audio1["input_values"].size()[0]
        audio1_encoding = self.audio_encoder(**audio1).last_hidden_state[:, -1, :]
        text1_encoding = self.text_encoder(**text1).last_hidden_state[:, -1, :]
        # print(text1_encoding.grad)

        audio2_encoding = self.audio_encoder(**audio2).last_hidden_state[:, -1, :]
        text2_encoding = self.text_encoder(**text2).last_hidden_state[:, -1, :]

        # print(audio1_encoding.requires_grad)

        hidden_vector = torch.cat(
            (
                audio1_encoding.reshape((batch_size, self.audio_hidden_size)),
                text1_encoding.reshape((batch_size, self.text_hidden_size)),
                audio2_encoding.reshape((batch_size, self.audio_hidden_size)),
                text2_encoding.reshape((batch_size, self.text_hidden_size)),
            ),
            dim=-1,
        )

        # print(hidden_vector.requires_grad)
        hidden_vector = self.dropout(hidden_vector)

        return self.head(hidden_vector)

    def freeze_encoders(self):
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = False

        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = False

    def unfreeze_encoders(self):
        for param in self.text_encoder.named_parameters():
            param[1].requires_grad = True

        for param in self.audio_encoder.named_parameters():
            param[1].requires_grad = True
