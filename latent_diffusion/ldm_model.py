import torch
from torch import nn
from utils.utils import get_grad_norm, AverageMeter, make_grid, noise_like, compute_grad_param_norms
from einops import rearrange, repeat
from tqdm import tqdm
import os


class LDM(nn.Module):
    def __init__(self, use_spatial_transformer=False,
                 scale_factor=0.18215,
                 model_logger_every=100,
                 diffusion_logger_every=50,
                 lr_anneal_steps=5e4,
                 init_lr=1e-5,
                 batch_size=64,
                 *,unet:nn.Module, vae:nn.Module, diffusion):
        super().__init__()
        self.vae = vae
        self.diffusion = diffusion
        self.unet = unet
        self.text_encoder = None
        self.lr = batch_size * init_lr
        # self.lr = init_lr
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        # self.lr_scheduler = LambdaLinearScheduler(self.optimizer)
        self.use_spatial_transformer = use_spatial_transformer
        self.scale_factor = scale_factor
        self.step = 0
        self.resume_step = 0
        self.model_logger_every = model_logger_every
        self.diffusion_logger_every = diffusion_logger_every
        self.lr_anneal_steps = lr_anneal_steps

        self.loss_meter = AverageMeter()
        self.loss_meter_val = AverageMeter()
        self.grad_norm_meter = AverageMeter()
        self.param_norm_meter = AverageMeter()
        self.best = 100


    def forward(self, x, c=None):
        # z = self.get_latent(x)
        # noise = torch.randn_like(z).to(z.device)
        # t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        # z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        # if self.use_spatial_transformer:
        #     assert c is not None, 'text condition can not be empty'
        #     c = self.get_text_condition(c)
        #
        # predicted_noise = self.unet(z_noisy, t, context=c)
        predicted_noise, target_noise = self._forward_unet(x, c)
        loss = self.get_loss(target_noise, predicted_noise)
        self.backprop(loss)
        self.anneal_lr()

    def _forward_unet(self, x, c=None):
        z = self.get_latent(x)
        noise = torch.randn_like(z).to(z.device)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        if self.use_spatial_transformer:
            assert c is not None, 'text condition can not be empty'
            c = self.get_text_condition(c)

        predicted_noise = self.unet(z_noisy, t, context=c)
        return predicted_noise, noise


    def backprop(self, loss):
        self.loss_meter.update(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        # grad_norm = get_grad_norm(self.unet.parameters())
        grad_norm, param_norm = compute_grad_param_norms(self.unet)
        self.grad_norm_meter.update(grad_norm)
        self.param_norm_meter.update(param_norm)
        self.optimizer.step()
        self.step += 1

    def anneal_lr(self,):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def validation(self, dataloader, logger, interval):
        if self.step % interval == 0:
            self.unet.eval()
            self.loss_meter_val.reset()
            for idx, batch in enumerate(dataloader):
                x = batch['image'].cuda()
                c = None # batch['c']
                predicted_noise, target_noise = self._forward_unet(x, c)
                loss = self.get_loss(target_noise, predicted_noise)
                self.loss_meter_val.update(loss.item())

            self.log_model(logger, 'validation')
            self.unet.train()

    def log_image(self, batch,
                  n_row=2, sample=True,
                  ddim_steps=None, ddim_eta=1.,
                  plot_denoise_rows=False,
                  plot_progressive_rows=False,
                  plot_diffusion_rows=False,
                  plot_reconstruction_rows=False,
                  return_input=False,):
        use_ddim = ddim_steps is not None
        _log = dict()

        x, c = batch['image'], batch['caption']
        x = x.cuda() # todo update here
        z = self.get_latent(x)
        _batch_size = z.shape[0]

        if self.use_spatial_transformer:
            c = self.get_text_condition(c)
        else:
            c = None


        if return_input:
            _log.update({'input': x})
        if plot_reconstruction_rows:
            reconstruction = self.get_image_from_latent(z)
            _log.update({'reconstruction': reconstruction})
        if plot_diffusion_rows:
            _diffused_images = list()
            z_start = z
            for t in range(self.diffusion.num_timesteps):
                if t % self.diffusion_logger_every == 0 or t == self.diffusion.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=_batch_size)
                    # t = t.to(self.device).long() #todo use to device
                    t = t.cuda().long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.diffusion.q_sample(x_start=z_start, t=t, noise=noise)
                    _diffused_images.append(self.get_image_from_latent(z_noisy))


            diffusion_row = torch.stack(_diffused_images)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            # _log["diffused_images"] = diffusion_grid
            _log.update({'diffused_images': diffusion_grid})
        if sample:
            # todo add later ema model
            samples, z_denoise_row = self.sample_log(c, _batch_size) # z_denoise_row is intermediates from T to 0
            x_samples = self.get_image_from_latent(samples)
            # _log["samples"] = x_samples
            _log.update({'samples': x_samples})
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                # _log["denoise_row"] = denoise_grid
                _log.update({'denoise_row': denoise_grid})

        if plot_progressive_rows:
            pass
            # todo to be implemented
            # # with self.ema_scope("Plotting Progressives"):
            # img, progressives = self.progressive_denoising(c, shape=(self.diffusion.channels, self.diffusion.image_size, self.diffusion.image_size), batch_size=_batch_size)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation Grid Handler")
            # # _log["progressive_row"] = prog_row
            # _log.update({'progressive_row': prog_row})

        return _log



    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        # for visualization purposes only
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.get_image_from_latent(zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim=None, ddim_steps=None, **kwargs):
        # todo ddim to be implemented
        # if ddim:
        #     ddim_sampler = DDIMSampler(self)
        #     shape = (self.channels, self.image_size, self.image_size)
        #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
        #                                                  shape, cond, verbose=False, **kwargs)
        #
        # else:
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def sample(self, cond, batch_size, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, x0=None, shape=None):

        if shape is None:
            shape = (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size)

        return self.p_sample_loop(shape, condition=cond, timesteps=timesteps, verbose=verbose, x0=x0, x_T=x_T, return_intermediates=return_intermediates)


    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None, timesteps=None, verbose=True, x_T=None, return_intermediates=False, x0=None, start_T=None, mask=None):
        device = self.diffusion.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.diffusion.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling <t>', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:  # TODO what is mask for
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, ts, condition=condition, clip_denoised=self.diffusion.clip_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % self.diffusion_logger_every == 0 or i == timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def p_sample(self, x, t, condition=None, clip_denoised=False, repeat_noise=False, return_x0=False, temperature=1., noise_dropout=0.):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x,
                                                                          c=condition,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised
                                                                          )

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, c, t,
                        clip_denoised: bool):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        based on models prediction of noise(aka eps), and given t, tries to predict the x_0, then computes the posterior of mu and var
        """

        model_output = self.unet(x, t, c)

        if self.diffusion.parameterization == 'eps':
            x_recon = self.diffusion.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            raise NotImplementedError('only epsilon prediction is implemented')
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon


    def log_model(self, logger, mode, writer=None):
            if writer is not None and mode=='train':
                writer.add_scalar(f'loss/train', self.loss_meter.val, self.step)
                writer.add_scalar(f'grad_norm/train', self.grad_norm_meter.val, self.step)
                writer.add_scalar(f'param_norm/train', self.param_norm_meter.val, self.step)

            if self.step % self.model_logger_every == 0:
                if mode == 'train':
                    lr = self.optimizer.param_groups[0]['lr']
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                    logger.info(f'{mode}/global_step:[{self.step}]\t'
                    f'loss {self.loss_meter.val:.5f} ({self.loss_meter.avg:.5f})\t'
                    f'grad_norm {self.grad_norm_meter.val:.5f} ({self.grad_norm_meter.avg:.5f})\t'
                    f'param_norm {self.param_norm_meter.val:.5f} ({self.param_norm_meter.avg:.5f})\t'
                    f'lr {lr:.9f} \t'
                    f'mem {memory_used:.2f}GB')
                elif mode=='validation':
                    logger.info(
                        f'{mode}/at step:[{self.step}]\t'
                        f'loss {self.loss_meter_val.val:.5f} ({self.loss_meter_val.avg:.5f})\t'
                    )
                    if self.best > self.loss_meter_val.avg:
                        self.best = self.loss_meter_val.avg
                        os.makedirs('checkpoint', exist_ok=True)
                        torch.save(self.unet.state_dict(), f'checkpoint/unet_{self.step}.pt')
                else:
                    raise ValueError



    def get_loss(self, target, prediction, loss_type='l2'):
        if loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(prediction, target)
        elif loss_type == 'l1':
            loss = (target - prediction).abs()
        else:
            raise NotImplementedError

        return loss


    @torch.no_grad()
    def get_latent(self, x):
        z = self.vae.encode(x).sample().detach()
        return z * self.scale_factor
    @torch.no_grad()
    def get_image_from_latent(self, z):
        z = 1. / self.scale_factor * z
        return self.vae.decode(z)

    def get_text_condition(self, c):
        return self.text_encoder(c)
