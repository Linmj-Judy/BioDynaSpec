import torch
import torch.nn as nn
import math

from .diffusion import create_diffusion, FrequencyComponentDenoiseNet

class SpectrumDiffLoss(nn.Module):
    """Spectrum Diffusion Loss

    参数:
        target_channels: int, 频谱通道数
        z_channels: int, 条件通道数
        depth: int, 网络深度
        width: int, 模型隐状态宽度
        num_sampling_steps: 生成采样步数（默认'100'字符串代表timesteps）
        learn_sigma: bool, 是否learned sigma（会自动设定denoise net out_channel*2）
        predict_xstart: bool, 是否模型直接预测clean x0，否则预测噪声
        sigma_small: bool, 用于扩散beta schedule
        noise_schedule: str, 扩散过程噪声类型
        use_kl: bool, 是否使用KL损失
        rescale_learned_sigmas: bool, 是否rescale learned sigma
        diffusion_steps: int, 总扩散步数
    """
    def __init__(self,
                 target_channels,
                 z_channels,
                 depth,
                 width,
                 num_sampling_steps='100',
                 learn_sigma=True,
                 predict_xstart=True,
                 sigma_small=False,
                 noise_schedule='cosine',
                 use_kl=False,
                 rescale_learned_sigmas=False,
                 diffusion_steps=1000,
                 denoise_seq_chunk_size=None,
                 denoise_block_type="residue",
                 denoise_use_checkpoint=False):
        super(SpectrumDiffLoss, self).__init__()
        self.in_channels = target_channels
        self.out_channels = target_channels * 2 if learn_sigma else target_channels

        self.net = FrequencyComponentDenoiseNet(
            in_channels=self.in_channels,
            model_channels=width,
            out_channels=self.out_channels,
            z_channels=z_channels,
            num_dit_blocks=depth,
            num_heads=16,
            mlp_ratio=4,
            seq_chunk_size=denoise_seq_chunk_size,
            block_type=denoise_block_type,
            use_checkpoint=denoise_use_checkpoint,
        ) # this is denoise net
        # 训练扩散
        self.train_diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            sigma_small=sigma_small,
            predict_xstart=predict_xstart,
            learn_sigma=learn_sigma,
            rescale_learned_sigmas=rescale_learned_sigmas,
            diffusion_steps=diffusion_steps
        )
        # 生成扩散
        self.gen_diffusion = create_diffusion(
            timestep_respacing=num_sampling_steps,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            sigma_small=sigma_small,
            predict_xstart=predict_xstart,
            learn_sigma=learn_sigma,
            rescale_learned_sigmas=rescale_learned_sigmas,
            diffusion_steps=diffusion_steps
        )

    def weight_loss(self, loss, loss_weight, seq_len):
        '''
        Weight the loss by the loss_weight
        Args:
            loss: (num_residues, )
            loss_weight: (seq_len, )
        Returns:
            标量loss
        ''' #TODO: 需要获得每个残基的loss再调整的话可以改这里的loss weight
        loss = loss.unsqueeze(-1).expand(-1, seq_len).contiguous() # (num_residues , seq_len)
        if loss_weight is not None:
            loss_weight = loss_weight.view(1, -1).contiguous() # (1, seq_len)
            loss_per_residue = (loss*loss_weight).sum(dim=1) / loss_weight.sum(dim=1).clamp(min=1e-6) # (num_residues, )
        else:
            loss_per_residue = loss # (num_residues, )
        return loss_per_residue.mean() # 标量

    def forward(self, target, z, index=None, mask=None, loss_weight=None, batch_mul: int = 1):
        """
        Input:
            target: (num_residues, seq_len, token_embed_dim) target spectrum
            z:      (num_residues, seq_len, decoder_embed_dim) condition from AR transformer
            index:  (num_residues, seq_len, 1)
            loss_weight: (seq_len, )
            batch_mul: 重复采样次数 M (即diffusion_batch_mul)
        Output:
            batch_pred_xstarts: (batch_mul, num_residues, seq_len, token_embed_dim)
            weighted_loss: 标量
        """
        num_residues, seq_len, _ = target.shape
        t = torch.randint(0, self.train_diffusion.num_timesteps, (1,), device=target.device) # 1：batch_size=1
        
        model_kwargs = dict(condition=z, index=index) # c: condition, index: stage index
        # 单次采样
        if batch_mul is None or batch_mul <= 1:
            loss_dict = self.train_diffusion.training_losses(model=self.net, x_start=target, t=t, model_kwargs=model_kwargs)
            pred_xstart = loss_dict["pred_xstart"] # shape: (num_residues, seq_len, token_embed_dim)
            
            loss = loss_dict["loss"]  # shape: (num_residues,)
            weighted_loss = self.weight_loss(loss, loss_weight, seq_len)
            return pred_xstart.unsqueeze(0), \
                weighted_loss
            
        # 多次采样（顺序求平均，显存≈单次）
        total_loss = 0.0
        batch_pred_xstarts = []
        for _ in range(batch_mul):
            t = torch.randint(0, self.train_diffusion.num_timesteps, (1,), device=target.device)
            loss_dict = self.train_diffusion.training_losses(model=self.net, x_start=target, t=t, model_kwargs=model_kwargs)
            batch_pred_xstarts.append(loss_dict["pred_xstart"])
            
            loss = loss_dict["loss"]  # shape: (num_residues,)
            sampled_loss = self.weight_loss(loss, loss_weight, seq_len)
            total_loss = total_loss + sampled_loss  # 逐样本累加
        weighted_loss = total_loss / float(batch_mul)
        batch_pred_xstarts = torch.stack(batch_pred_xstarts, dim=0) # shape: (batch_mul, num_residues, seq_len, token_embed_dim)
        return batch_pred_xstarts, weighted_loss
    
    def sample(self, z, temperature=1.0, cfg=1.0, index=None):
        # diffusion loss sampling
        # z shape: (2*num_residues, seq_len, decoder_embed_dim) for CFG
        # Generate noise with shape (num_residues, seq_len, in_channels)
        noise = torch.randn(z.shape[0] // 2, z.shape[1], self.in_channels).cuda() # (num_residues, seq_len, in_channels)
        noise = torch.cat([noise, noise], dim=0)  # (2*num_residues, seq_len, in_channels)
        model_kwargs = dict(condition=z, cfg_scale=cfg, index=index)
        sample_fn = self.net.forward_with_cfg

        # Use clip_denoised=False but provide custom denoised_fn to clip to proper range
        # Training data spectrum values are in range [-1.75, 1.75]
        # We add margin and clip to [-2.0, 2.0] to avoid cutting off valid values
        def denoised_fn(x):
            # return torch.clamp(x, min=-2.0, max=2.0)
            return x

        # DDIM sampler, as described in the paper (Sec. 4.2): 50 respaced steps
        # via Denoising Diffusion Implicit Models. We use eta=0, i.e. the
        # deterministic DDIM update.
        #
        # NOTE: DDIM has no temperature knob. p_sample_loop() used to scale the
        # injected noise by `temperature`, but that path is ancestral sampling
        # and does not match the paper. temperature is therefore only applied
        # when it differs from the default, as stochasticity (eta) — a
        # temperature < 1 is not representable without changing the sampler, so
        # we guard against silently ignoring it.
        eta = 0.0
        if temperature != 1.0:
            raise ValueError(
                "The DDIM sampler (paper Sec. 4.2) is deterministic and does not "
                f"support temperature != 1.0 (got {temperature}). "
                "Use eta>0 in ddim_sample if stochastic DDIM is intended."
            )

        sampled_spectrum, pred_start_spectrum = self.gen_diffusion.ddim_sample_loop(
            sample_fn, noise.shape, noise,
            clip_denoised=False,  # Don't use built-in [-1,1] clipping
            denoised_fn=denoised_fn,  # Use custom clipping range
            model_kwargs=model_kwargs,
            progress=False,
            eta=eta,
        )

        return sampled_spectrum, pred_start_spectrum
