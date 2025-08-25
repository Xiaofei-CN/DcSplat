from einops import rearrange
from typing import Any, Dict
from diffusers.models.attention import AdaLayerNorm, AdaLayerNormZero, FeedForward
from diffusers.models.attention_processor import *
from models.baseFunction import ResidualBlock

class BasicTransBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            cross_attention_dim2: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_instance_norm = norm_type == "instance_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif self.use_instance_norm:
            self.norm1 = nn.InstanceNorm1d(1024)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm2 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif self.use_instance_norm:
                self.norm2 = nn.InstanceNorm1d(1024)
            else:
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        if cross_attention_dim2 is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm4 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm4 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif self.use_instance_norm:
                self.norm4 = nn.InstanceNorm1d(1024)
            else:
                self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn3 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm4 = None
            self.attn3 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            query_pos: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            other_encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states + query_pos
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_instance_norm:
                norm_hidden_states = self.norm2(rearrange(hidden_states, "b d h -> b h d"))
                norm_hidden_states = rearrange(norm_hidden_states, "b h d -> b d h")
            else:
                norm_hidden_states = self.norm2(hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        if self.attn3 is not None:
            hidden_states = hidden_states + query_pos
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm4(hidden_states, timestep)
            elif self.use_instance_norm:
                norm_hidden_states = self.norm4(rearrange(hidden_states, "b d h -> b h d"))
                norm_hidden_states = rearrange(norm_hidden_states, "b h d -> b d h")
            else:
                norm_hidden_states = self.norm4(hidden_states)

            attn_output = self.attn3(
                norm_hidden_states,
                encoder_hidden_states=other_encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_instance_norm:
            norm_hidden_states = self.norm1(rearrange(hidden_states, "b d h -> b h d"))
            norm_hidden_states = rearrange(norm_hidden_states, "b h d -> b d h")
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class BasicTransBlock_in(nn.Module):

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            cross_attention_dim2: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_instance_norm = norm_type == "instance_norm"

        if norm_type in ("ada_norm", "ada_norm_zero","instance_norm") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif self.use_instance_norm:
            self.norm1 = nn.InstanceNorm1d(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm2 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif self.use_instance_norm:
                self.norm2 = nn.InstanceNorm1d(dim)
            else:
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        if cross_attention_dim2 is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm4 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm4 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif self.use_instance_norm:
                self.norm4 = nn.InstanceNorm1d(dim)
            else:
                self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

            self.attn3 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm4 = None
            self.attn3 = None

        # 3. Feed-forward
        self.norm3 = nn.InstanceNorm1d(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            query_pos: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            other_encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        if self.attn3 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else self.norm4(hidden_states)
            )

            attn_output = self.attn3(
                norm_hidden_states,
                encoder_hidden_states=other_encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class Decoder(nn.Module):
    """
    Parameters:108 584 960
    """

    def __init__(self, n_ctx=64, ctx_dim=768, heads=24, depth=8,norm_type="layer_norm"):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.depth = depth

        self.kv_pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.kv_pos_embed, std=0.02)
        self.kv_pos_embed_copy = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.kv_pos_embed_copy, std=0.02)

        self.q_pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.q_pos_embed, std=0.02)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(BasicTransBlock(
                dim=ctx_dim,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                cross_attention_dim=ctx_dim,
                cross_attention_dim2=ctx_dim,
                norm_type=norm_type#"instance_norm"
            ))
        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, other_encoder_hidden_states):
        B, L, C = hidden_states.shape
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
        kv_pos_embed_copy = self.kv_pos_embed_copy.expand(B, -1, -1)
        pos_embed = self.q_pos_embed.expand(B, -1, -1)
        encoder_hidden_states = encoder_hidden_states + kv_pos_embed
        other_encoder_hidden_states = other_encoder_hidden_states + kv_pos_embed_copy

        for blk in self.blocks:
            hidden_states = blk(hidden_states, pos_embed,
                                encoder_hidden_states=encoder_hidden_states,
                                other_encoder_hidden_states=other_encoder_hidden_states)
        return hidden_states
class Encoder(nn.Module):
    """
    Parameters:108 584 960
    """

    def __init__(self, n_ctx=64, ctx_dim=768, heads=24, depth=8,norm_type="layer_norm"):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.depth = depth

        self.q_pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.q_pos_embed, std=0.02)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(BasicTransBlock(
                dim=ctx_dim,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                norm_type=norm_type,#"instance_norm"
            ))
        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states):
        B, L, C = hidden_states.shape
        pos_embed = self.q_pos_embed.expand(B, -1, -1)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, pos_embed)
        return hidden_states


class SideDecoder(nn.Module):

    def __init__(self, n_ctx=64, ctx_dim=768, heads=24,depth=8):
        super().__init__() #instance_norm  layer_norm
        norm_type = "instance_norm"
        self.decoder = Decoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)
        self.fs_encoder = Encoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)
        self.fd_encoder = Encoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)

    def forward(self, fs, fp, fd=None,v=4):
        _,_,h,w = fs.shape
        fs = rearrange(fs, "b c h w -> b (h w) c")
        fs_enhance = self.fs_encoder(fs)

        if fd is not None and self.fd_encoder:
            fd = rearrange(fd, "b c h w -> b (h w) c")
            fd_enhance = self.fd_encoder(fd)

        fp = rearrange(fp, "(b v) c h w -> b v (h w) c", v=v)
        fd_enhance = rearrange(fd_enhance, "(b v) c d -> b v c d", v=v)
        out = [fs_enhance]
        for i in range(1,v):
            side_img_feat = self.decoder(fp[:, i, ...], fs_enhance, fd_enhance[:,i,...])
            out.append(side_img_feat)

        out = torch.stack(out, dim=1) #b v n c
        out = rearrange(out,"b v (h w) c -> (b v) c h w", v=v,h=h, w=w)
        return out

class Ablation_SideDecoderWoEncoder(nn.Module):

    def __init__(self, n_ctx=64, ctx_dim=768, heads=24,depth=8):
        super().__init__() #instance_norm  layer_norm
        norm_type = "instance_norm"
        self.decoder = Decoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)

    def forward(self, fs, fp, fd=None,v=4):
        _,_,h,w = fs.shape
        fs_enhance = rearrange(fs, "b c h w -> b (h w) c")
        fd_enhance = rearrange(fd, "(b v) c h w -> b v (h w) c",v=v)

        fp = rearrange(fp, "(b v) c h w -> b v (h w) c", v=v)
        out = [fs_enhance]
        for i in range(1,v):
            side_img_feat = self.decoder(fp[:, i, ...], fs_enhance, fd_enhance[:,i,...])
            out.append(side_img_feat)

        out = torch.stack(out, dim=1) #b v n c
        out = rearrange(out,"b v (h w) c -> (b v) c h w", v=v,h=h, w=w)
        return out
class Ablation_CNNDeocder(nn.Module):
    def __init__(self,n_ctx=64, ctx_dim=768, heads=24,depth=8,norm_fn='group'):
        super().__init__()
        # self.decoder = Decoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads)
        self.decoder = nn.Sequential(

            ResidualBlock(256*3, 256*2, norm_fn=norm_fn),
            ResidualBlock(256*2, 256, norm_fn=norm_fn),

            )
        # self.fs_encoder = nn.Sequential(
        #
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #
        #     )
        norm_type = "instance_norm"
        self.fs_encoder = Encoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)
        self.fd_encoder = Encoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)

    def forward(self, fs, fp, fd=None,v=4):
        b,c,h,w = fs.shape
        fs = rearrange(fs, "b c h w -> b (h w) c")
        fs_enhance = self.fs_encoder(fs)
        fs_enhance = rearrange(fs_enhance, "b (h w) c -> b c h w",b=b,c=c,h=h,w=w)

        if fd is not None and self.fd_encoder is not None:
            fd = rearrange(fd, "b c h w -> b (h w) c")
            fd_enhance = self.fd_encoder(fd)
            fd_enhance = rearrange(fd_enhance, "(b v) (h w) c -> b v c h w", v=v,c=c, h=h, w=w)
        fp = rearrange(fp, "(b v) c h w -> b v c h w", v=v)

        out = [fs_enhance]
        for i in range(1, v):
            side_img_feat = \
            self.decoder(torch.cat([fp[:, i, ...], fs_enhance, fd_enhance[:, i, ...]], dim=1))
            out.append(side_img_feat)

        out = torch.stack(out, dim=1)  # b v n c
        out = rearrange(out,"b v c h w -> (b v) c h w")
        return out
class Ablation_CNNDeocderWE1(nn.Module):
    def __init__(self,n_ctx=64, ctx_dim=768, heads=24,depth=8,norm_fn='group'):
        super().__init__()
        # self.decoder = Decoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads)
        self.decoder = nn.Sequential(

            ResidualBlock(256*3, 256*2, norm_fn=norm_fn),
            ResidualBlock(256*2, 256, norm_fn=norm_fn),

            )
        # self.fs_encoder = nn.Sequential(
        #
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #     ResidualBlock(32, 32, norm_fn=norm_fn),
        #
        #     )
        norm_type = "instance_norm"
        self.fs_encoder = Encoder(n_ctx=n_ctx, ctx_dim=ctx_dim, depth=depth, heads=heads,norm_type=norm_type)

    def forward(self, fs, fp, fd=None,v=4):
        b,c,h,w = fs.shape
        fs = rearrange(fs, "b c h w -> b (h w) c")
        fs_enhance = self.fs_encoder(fs)
        fs_enhance = rearrange(fs_enhance, "b (h w) c -> b c h w",b=b,c=c,h=h,w=w)

        fd_enhance = rearrange(fd, "(b v) c h w -> b v c h w", v=v,c=c, h=h, w=w)
        fp = rearrange(fp, "(b v) c h w -> b v c h w", v=v)

        out = [fs_enhance]
        for i in range(1, v):
            side_img_feat = \
                self.decoder(torch.cat([fp[:, i, ...], fs_enhance, fd_enhance[:, i, ...]], dim=1))
            out.append(side_img_feat)

        out = torch.stack(out, dim=1)  # b v n c
        out = rearrange(out,"b v c h w -> (b v) c h w")
        return out
class Ablation_CNNDeocderWoEncoder(nn.Module):
    def __init__(self,n_ctx=64, ctx_dim=768, heads=24,depth=8,norm_fn='group'):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualBlock(256*3, 256*2, norm_fn=norm_fn),
            ResidualBlock(256*2, 256, norm_fn=norm_fn),
            )

    def forward(self, fs, fp, fd=None,v=4):
        b,c,h,w = fs.shape
        fs_enhance = fs
        fd_enhance = rearrange(fd, "(b v) c h w -> b v c h w", v=v,c=c, h=h, w=w)
        fp = rearrange(fp, "(b v) c h w -> b v c h w", v=v)

        out = [fs_enhance]
        for i in range(1, v):
            side_img_feat = \
                self.decoder(torch.cat([fp[:, i, ...], fs_enhance, fd_enhance[:, i, ...]], dim=1))
            out.append(side_img_feat)

        out = torch.stack(out, dim=1)  # b v n c
        out = rearrange(out,"b v c h w -> (b v) c h w")
        return out