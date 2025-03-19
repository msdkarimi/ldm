from configs.train_model_config import get_first_stage_config, get_diffusion_config, get_unet_config,get_latent_diffusion_config
from variational_ae.autoencoder import AutoencoderKL
from gassian_diffusion.diffusion import GaussianDiffusion
import torch
from latent_diffusion.ldm_model import LDM
from unet.openai_model import UNetModel
from data.data_loader import DareDataset
from torch.utils.data import DataLoader
from utils.utils import image_transform, count_params
from utils.logger import build_logger
from utils.image_logger import ImageLogger
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = 'ldm_logger'
BATCH_SIZE = 64
MAX_STEP = 5e5
VALIDATION_EVERY = 1e3

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = AutoencoderKL(**get_first_stage_config())
    gaussian_diffusion = GaussianDiffusion(**get_diffusion_config())
    unet = UNetModel(**get_unet_config())
    count_params(unet, True)
    ldm_model = LDM(**get_latent_diffusion_config(), unet=unet , vae=vae, diffusion=gaussian_diffusion).to(device)

    train_dataset = DareDataset('data', 'train', transform=image_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    val_dataset = DareDataset('data', 'validation', transform=image_transform)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    logger, logger_name = build_logger(LOG_DIR, 'model_logger')
    writer = SummaryWriter(f'{LOG_DIR}/{logger_name}/run')
    image_logger = ImageLogger(LOG_DIR, logger_name,
                               {
                                   'n_row':2, 'sample':True,
                                   'ddim_steps':None, 'ddim_eta':1.,
                                   'plot_reconstruction_rows':True, 'plot_denoise_rows':True,
                                   'plot_progressive_rows':False, 'plot_diffusion_rows':True, 'return_input':True
                               })

    logger.info('init train')

    while ldm_model.step < MAX_STEP:

        try:
            batch = next(iter(train_data_loader))
            x = batch['image'].to(device)
            caption = batch['caption']
        except StopIteration:
            train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
            batch = next(iter(train_data_loader))
            x = batch['image'].to(device)
            caption = batch['caption']

        ldm_model(x, c=None)
        ldm_model.log_model(logger, 'train', writer)
        image_logger.do_log(ldm_model, 'train', batch, ldm_model.step)
        ldm_model.validation(val_data_loader, logger, VALIDATION_EVERY)




if __name__ == '__main__':
    main()
    exit(0)