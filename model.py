import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import transformers


class ClassificationHead(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, n_classes)

        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])

        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.output.weight)

        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        out = F.tanh(self.input(x))

        for layer in self.hidden:
            out = F.tanh(layer(out))

        out = self.output(out)

        return out


class ConcatModel(nn.Module):
    def __init__(
        self,
        text_encoder_checkpoint,
        audio_encoder_checkpoint,
        text_hidden_size,
        audio_hidden_size,
        n_classes=4,
        dropout=0.5,
    ):
        super().__init__()
        self.text_encoder = transformers.BertModel.from_pretrained(
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

        self.text_dropout = nn.Dropout(p=dropout)
        self.audio_dropout = nn.Dropout(p=dropout)

        self.head = ClassificationHead(
            text_hidden_size * 2 + audio_hidden_size * 2, n_classes
        )

        # self.freeze_encoders()
        self.unfreeze_encoders()

    def get_encoding(self, audio, text):
        text_encoding_pooled = self.text_encoder(**text)[1]
        text_encoding_pooled = self.text_dropout(text_encoding_pooled)

        # print(text_encoding.shape)
        # print(text["attention_mask"].shape)

        # text_encoding_pooled = (text_encoding * text["attention_mask"][:, :, None]).sum(dim=1)
        # text_encoding_pooled = text_encoding / text["attention_mask"].sum(dim=1)[:, None]
        # text_encoding_pooled = text_encoding.mean(dim=1)

        audio_encoding = self.audio_encoder(**audio)[0]
        audio_encoding = self.audio_dropout(audio_encoding)

        audio_encoding_pooled = audio_encoding.mean(dim=1)

        # print(audio_encoding.shape)
        # print(audio_encoding_pooled.shape)
        # audio_encoding = audio_encoding / audio["attention_mask"].sum(dim=1)[:, None]

        concat_encoding = torch.cat(
            (text_encoding_pooled, audio_encoding_pooled), dim=-1
        )

        return concat_encoding

    def forward(self, audio1, text1, audio2, text2, **kwargs):
        # print(audio1)
        # print(text1)
        seq1_encoding = self.get_encoding(audio1, text1)
        seq2_encoding = self.get_encoding(audio2, text2)

        hidden_vector = torch.cat(
            (seq1_encoding, seq2_encoding),
            dim=-1,
        )

        # hidden_vector = self.get_encoding(text)

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
