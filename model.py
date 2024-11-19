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
    ):
        super().__init__()
        self.text_encoder = transformers.AutoModel.from_pretrained(
            text_encoder_checkpoint
        )
        self.audio_encoder = transformers.AutoModel.from_pretrained(
            audio_encoder_checkpoint
        )

        self.text_hidden_size = text_hidden_size
        self.audio_hidden_size = audio_hidden_size

        self.head = ClassificationHead(
            text_hidden_size * 2 + audio_hidden_size * 2, n_classes
        )

    def forward(self, audio1, text1, audio2, text2, **kwargs):
        batch_size = audio1["input_values"].size()[0]
        audio1_encoding = self.audio_encoder(**audio1).last_hidden_state
        text1_encoding = self.text_encoder(**text1).last_hidden_state

        audio2_encoding = self.audio_encoder(**audio2).last_hidden_state
        text2_encoding = self.text_encoder(**text2).last_hidden_state

        print(audio1_encoding.requires_grad)

        hidden_vector = torch.cat(
            (
                audio1_encoding.view((batch_size, self.audio_hidden_size)),
                text1_encoding.view((batch_size, self.text_hidden_size)),
                audio2_encoding.view((batch_size, self.audio_hidden_size)),
                text2_encoding.view((batch_size, self.text_hidden_size)),
            ),
            dim=1,
        )

        print(hidden_vector.requires_grad)

        return self.head(hidden_vector)
