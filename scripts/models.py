import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtV2Block1D(nn.Module):
    """ ConvNeXtV2 Block for 1D sequences.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.grn = GRN(4 * dim) # GRN not used in this version
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)
        return x 


class Encoder(nn.Module):
    def __init__(self, in_channels=513, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], drop_path_rate=0.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block1D(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=513, depth=1):
        super().__init__()
        self.decoder = nn.Sequential(
            *[ConvNeXtV2Block1D(dim=in_channels) for _ in range(depth)]
        )
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.decoder(x)
        x = self.proj(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, in_channels=768, hidden_dim=1536, out_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x):
        # x is (N, L, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # output is (N, L, out_channels)
        return x

class WaveMAE(nn.Module):
    def __init__(self, in_channels=513, encoder_dims=[96, 192, 384, 768], encoder_depths=[3, 3, 9, 3], 
                 decoder_depth=1, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.encoder = Encoder(in_channels=in_channels, dims=encoder_dims, depths=encoder_depths)
        
        # The decoder input dimension should match the last dimension of the encoder
        decoder_in_channels = encoder_dims[-1]
        self.decoder = Decoder(in_channels=decoder_in_channels, out_channels=in_channels, depth=decoder_depth)

        self.crepe_decoder = MLPDecoder(in_channels=decoder_in_channels, out_channels=1)
        # As per plan, w2v-bert is disabled for now.
        # self.w2v_bert_decoder = MLPDecoder(in_channels=decoder_in_channels, out_channels=768)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_in_channels))
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, stft, crepe_pitch):
        B, C, L = stft.shape

        # Masking
        num_masked = int(self.mask_ratio * L)
        noise = torch.rand(B, L, device=stft.device)
        masked_indices = noise.topk(num_masked, dim=-1).indices
        
        # We create a boolean mask for convenience
        mask = torch.ones(B, L, device=stft.device, dtype=torch.bool)
        mask.scatter_(1, masked_indices, False) # False for masked positions

        # For now, we use a simple (but inefficient) masking approach for the encoder
        # This can be optimized later to only process unmasked tokens.
        unmasked_stft = stft * mask.unsqueeze(1).float()
        
        latent_representation = self.encoder(unmasked_stft) # B, D, L_downsampled

        # Upsample latent representation to original sequence length
        latent_representation_upsampled = F.interpolate(latent_representation, size=L, mode='linear') # B, D, L
        
        # Prepare for decoder: add mask tokens
        mask_tokens = self.mask_token.expand(B, L, -1).permute(0, 2, 1) # B, D, L
        
        # Replace masked parts with mask tokens
        # We need to expand the mask to match the latent dim
        # mask is (B, L), we need (B, D, L)
        expanded_mask = mask.unsqueeze(1).expand_as(latent_representation_upsampled)
        decoder_input = torch.where(expanded_mask, latent_representation_upsampled, mask_tokens)

        # Main decoder
        reconstructed_stft = self.decoder(decoder_input)

        # Auxiliary decoders
        # The decoders operate on the latent representation of the masked parts
        # We need to gather the latent vectors at the masked positions.
        latent_transposed = latent_representation_upsampled.permute(0, 2, 1) # B, L, D
        
        masked_latent_vectors = torch.zeros(B, num_masked, latent_transposed.size(2), device=stft.device)
        for i in range(B):
            masked_latent_vectors[i] = latent_transposed[i, masked_indices[i]]
            
        predicted_pitch = self.crepe_decoder(masked_latent_vectors) # (B, num_masked, 1)
        
        # For now, predicted_w2v_bert is None
        predicted_w2v_bert = None

        return reconstructed_stft, predicted_pitch, predicted_w2v_bert, ~mask # return inverse mask (1 for masked)

    def patchify(self, x):
        # This model works on frames, so patchify is not needed in the same way as for images.
        pass

    def unpatchify(self, x):
        pass 