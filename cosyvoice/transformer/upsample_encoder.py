# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from cosyvoice.transformer.convolution import ConvolutionModule
from cosyvoice.transformer.embedding import PositionalEncoding
from cosyvoice.transformer.encoder_layer import ConformerEncoderLayer
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.mask import add_optional_chunk_mask


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        else:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(torch.concat([outputs, context], dim=2), (0, self.pre_lookahead_len - context.size(2)), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
        num_prosody_tokens: int = 0,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)
        self.pre_lookahead_layer = PreLookaheadLayer(in_channels=512, channels=512, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(4)
        ])
        # optional prosody cross-attention (one sublayer per encoder block)
        # num_prosody_tokens > 0  → option 2: discrete token Embedding
        # (call init_prosody_encoder() separately for option 3: continuous features)
        if num_prosody_tokens > 0:
            self.prosody_embedding = nn.Embedding(num_prosody_tokens, output_size)
            self.prosody_pos_enc = PositionalEncoding(output_size, dropout_rate)
            # pre-upsampler: one per encoders block (num_blocks)
            self.prosody_cross_attn = nn.ModuleList([
                nn.MultiheadAttention(output_size, attention_heads,
                                      dropout=attention_dropout_rate, batch_first=True)
                for _ in range(num_blocks)
            ])
            self.prosody_cross_attn_norm = nn.ModuleList([
                nn.LayerNorm(output_size, eps=1e-5) for _ in range(num_blocks)
            ])
            # post-upsampler: one per up_encoders block (4)
            self.prosody_up_cross_attn = nn.ModuleList([
                nn.MultiheadAttention(output_size, attention_heads,
                                      dropout=attention_dropout_rate, batch_first=True)
                for _ in range(4)
            ])
            self.prosody_up_cross_attn_norm = nn.ModuleList([
                nn.LayerNorm(output_size, eps=1e-5) for _ in range(4)
            ])

    def output_size(self) -> int:
        return self._output_size

    def init_prosody_encoder(self, prosody_encoder_path: str, prosody_feat_dim: int = 768,
                              prosody_pool_factor: int = 5):
        """Load a frozen ProsodyEncoder for continuous prosody feature extraction (option 3).

        Creates self.prosody_proj (Linear) and self.prosody_encoder (frozen).
        A prosody_pos_enc is also created if not already present.
        """
        from cosyvoice.prosody.prosody_encoder import ProsodyEncoder
        output_size = self._output_size
        self.prosody_encoder = ProsodyEncoder.from_checkpoint(prosody_encoder_path)
        for p in self.prosody_encoder.parameters():
            p.requires_grad_(False)
        self.prosody_proj = nn.Linear(prosody_feat_dim, output_size)
        self.prosody_pool_factor = prosody_pool_factor
        if not hasattr(self, 'prosody_pos_enc'):
            self.prosody_pos_enc = PositionalEncoding(output_size, 0.0)

    def extract_prosody_emb(self, glottal_16k: torch.Tensor,
                             glottal_16k_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract and project continuous prosody features from 16 kHz glottal source.

        Returns [B, T', output_size] at ~12.5 Hz (after pool_factor downsampling).
        """
        padding_mask = None
        if glottal_16k_len is not None:
            max_len = glottal_16k.shape[1]
            padding_mask = (torch.arange(max_len, device=glottal_16k.device).unsqueeze(0)
                            >= glottal_16k_len.unsqueeze(1))
        with torch.no_grad():
            feats, _ = self.prosody_encoder(glottal_16k, padding_mask)  # [B, T', 768]
        if self.prosody_pool_factor > 1:
            B, T, D = feats.shape
            T_out = T // self.prosody_pool_factor
            feats = feats[:, :T_out * self.prosody_pool_factor, :].view(
                B, T_out, self.prosody_pool_factor, D).mean(dim=2)
        return self.prosody_proj(feats)  # [B, T', output_size]

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        context: torch.Tensor = torch.zeros(0, 0, 0),
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
        prosody_token: Optional[torch.Tensor] = None,
        prosody_token_len: Optional[torch.Tensor] = None,
        glottal_16k: Optional[torch.Tensor] = None,
        glottal_16k_len: Optional[torch.Tensor] = None,
        prosody_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        if context.size(1) != 0:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            context_masks = torch.ones(1, 1, context.size(1)).to(masks)
            context, _, _ = self.embed(context, context_masks, offset=xs.size(1))
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks, False, False, 0, self.static_chunk_size if streaming is True else 0, -1)
        # compute prosody embeddings once for use in both encoder stages
        # pre-computed > option 3 (continuous) > option 2 (discrete tokens)
        prosody_key_padding_mask = None
        if prosody_emb is not None:
            prosody_emb, _ = self.prosody_pos_enc(prosody_emb.to(xs.device))  # already encoded, just add pos enc
        elif glottal_16k is not None and hasattr(self, 'prosody_encoder'):
            raw_emb = self.extract_prosody_emb(glottal_16k, glottal_16k_len)  # [B, T', D]
            prosody_emb, _ = self.prosody_pos_enc(raw_emb)
            # no explicit padding mask needed (extract_prosody_emb returns valid frames only)
        elif prosody_token is not None and hasattr(self, 'prosody_embedding'):
            prosody_emb, _ = self.prosody_pos_enc(self.prosody_embedding(prosody_token))  # [B, L_p, D]
            if prosody_token_len is not None:
                prosody_key_padding_mask = make_pad_mask(prosody_token_len)  # [B, L_p], True=padding

        # lookahead + conformer encoder (pre-upsampler)
        xs = self.pre_lookahead_layer(xs, context=context)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad, prosody_emb, prosody_key_padding_mask)

        # upsample + conformer encoder (post-upsampler)
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks, False, False, 0, self.static_chunk_size * self.up_layer.stride if streaming is True else 0, -1)
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad, prosody_emb, prosody_key_padding_mask)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor, mask_pad: torch.Tensor,
                       prosody_emb: torch.Tensor = None,
                       prosody_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for i, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            if prosody_emb is not None and hasattr(self, 'prosody_cross_attn'):
                residual = xs
                xs_norm = self.prosody_cross_attn_norm[i](xs)
                xs_ca, _ = self.prosody_cross_attn[i](
                    query=xs_norm, key=prosody_emb, value=prosody_emb,
                    key_padding_mask=prosody_key_padding_mask,
                )
                xs = residual + xs_ca
        return xs

    def forward_up_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                          pos_emb: torch.Tensor, mask_pad: torch.Tensor,
                          prosody_emb: torch.Tensor = None,
                          prosody_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for i, layer in enumerate(self.up_encoders):
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            if prosody_emb is not None and hasattr(self, 'prosody_up_cross_attn'):
                residual = xs
                xs_norm = self.prosody_up_cross_attn_norm[i](xs)
                xs_ca, _ = self.prosody_up_cross_attn[i](
                    query=xs_norm, key=prosody_emb, value=prosody_emb,
                    key_padding_mask=prosody_key_padding_mask,
                )
                xs = residual + xs_ca
        return xs
