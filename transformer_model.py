import torch
import torch.nn as nn
from ..layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import DataEmbedding
from st40.infra.torch import *
from scipy import stats

class TransformerModel(BaseModule):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 8,
        nhead: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 1,
        dropout: float = 0.5,
        label_len: int= 1,
        lossfunc: str = 'wmse',
    ):
        super().__init__()

        self.transformer = Transformer(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=output_dim,
            d_model=d_model,
            n_heads=nhead,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            label_len=label_len,
        )
        self.criterion = get_lossfunc(lossfunc)
        self.d_model = d_model

    def forward(self, x):
        return self.transformer(x)

    def train_step(self, batch_ndx, batch, device):
        y, x, w = batch
        y, x, w = y.to(device), x.to(device), w.to(device)
        pred = self(x)
        pearson_corr = torch.tensor(stats.pearsonr(pred.detach().cpu(), y.detach().cpu())[0])
        spearman_corr = torch.tensor(stats.spearmanr(pred.detach().cpu(), y.detach().cpu())[0])
        if torch.isnan(pearson_corr) or torch.isnan(spearman_corr):
            print(pred)
            raise ValueError("The model might be overfitting. Pearson or Spearman correlation is NaN.")
        return self.criterion(pred,y,w), pearson_corr, spearman_corr

    def valid_step(self, batch_ndx, batch, device):
        with torch.no_grad():
            y, x, w = batch
            y, x, w = y.to(device), x.to(device), w.to(device)
            pred = self(x)
            pearson_corr = torch.tensor(stats.pearsonr(pred.detach().cpu(), y.detach().cpu())[0])
            spearman_corr = torch.tensor(stats.spearmanr(pred.detach().cpu(), y.detach().cpu())[0])
            if torch.isnan(pearson_corr) or torch.isnan(spearman_corr):
                print(pred)
                raise ValueError("The model might be overfitting. Pearson or Spearman correlation is NaN.")
        return self.criterion(pred,y,w), pearson_corr, spearman_corr

    def test_step(self, batch_ndx, batch, device):
        with torch.no_grad():
            y, x, w = batch
            y, x, w = y.to(device), x.to(device), w.to(device)
            pred = self(x)
            pearson_corr = torch.tensor(stats.pearsonr(pred.detach().cpu(), y.detach().cpu())[0])
            spearman_corr = torch.tensor(stats.spearmanr(pred.detach().cpu(), y.detach().cpu())[0])
            if torch.isnan(pearson_corr) or torch.isnan(spearman_corr):
                print(pred)
                raise ValueError("The model might be overfitting. Pearson or Spearman correlation is NaN.")
        return self.criterion(pred,y,w), pearson_corr, spearman_corr


class Transformer(nn.Module):
    """
    Vanilla Transformer with full encoder and decoder
    """
    def __init__(self,
                enc_in=8,
                dec_in=8,
                c_out=1,
                d_model=8,
                n_heads=8,
                e_layers=2,
                d_layers=1,
                d_ff=1,
                freq='h',
                factor=1,
                dropout=0.5,
                embed='fixed',
                activation='gelu',
                output_attention=False,
                label_len=1,
                ):
        super().__init__()
        self.label_len = label_len

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x):

        # construct inputs
        x_enc = x
        x_mark_enc = None
        x_dec = x[:, -self.label_len:, :]
        x_mark_dec = None

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out[:,-1, :].squeeze() # [B, L, D]
