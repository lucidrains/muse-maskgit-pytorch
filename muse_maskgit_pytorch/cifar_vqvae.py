"""
CIFAR10 vqvae from https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical
import math


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.999,
        epsilon=1e-5,
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward_from_indices(self, indices):
        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(
            self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
                encodings, dim=1
            )

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (
                (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            )

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        )

        return (
            quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W),
            loss,
            perplexity.sum(),
            indices,
        )

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(
            torch.sum(self.embedding**2, dim=2).unsqueeze(1)
            + torch.sum(x_flat**2, dim=2, keepdim=True),
            x_flat,
            self.embedding.transpose(1, 2),
            alpha=-2.0,
            beta=1.0,
        )

        indices = torch.argmin(distances, dim=-1)
        return self.forward_from_indices(indices)


class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(
            torch.Tensor(latent_dim, num_embeddings, embedding_dim)
        )
        nn.init.uniform_(self.embedding, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(
            torch.sum(self.embedding**2, dim=2).unsqueeze(1)
            + torch.sum(x_flat**2, dim=2, keepdim=True),
            x_flat,
            self.embedding.transpose(1, 2),
            alpha=-2.0,
            beta=1.0,
        )
        distances = distances.view(N, B, H, W, M)

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        if self.training:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        )

        return (
            quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W),
            KL,
            perplexity.sum(),
        )


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 3 * 256, 1),
        )

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist


class CIFARVQVAE(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim):
        super(CIFARVQVAE, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, loss, perplexity, indices = self.codebook(x)
        dist = self.decoder(x)
        return dist, loss, perplexity, indices


class GSSOFT(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim):
        super(GSSOFT, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, KL, perplexity = self.codebook(x)
        dist = self.decoder(x)
        return dist, KL, perplexity
