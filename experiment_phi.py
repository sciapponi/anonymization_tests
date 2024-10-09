import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch 
from torch import nn
import torch.nn.functional as F
from itertools import chain
import wandb
from discriminators import WaveDiscriminator, STFTDiscriminator
from losses import ReconstructionLoss, ReconstructionLoss2#, XVectorLoss
# from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from pesq import pesq
from model import SoundPhi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio import  ScaleInvariantSignalNoiseRatio as SISNR
import hydra

torch.set_float32_matmul_precision('medium')

# if torch.cuda.is_available(): 
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device('cuda')

class ExperimentPhi(L.LightningModule):

    def __init__(self, 
                 args,):
        
        super().__init__()

        self.args=args
        self.dataset_args = args.dataset_args
        
        self.automatic_optimization = False

        self.model = SoundPhi(latent_space_dim=args.model.latent_space_dim,
                              n_q=16,
                              codebook_size=1024)
        

        # DISCRIMINATORS
        if args.losses.discriminators:
            self.wave_discriminators = nn.ModuleList([
                WaveDiscriminator(resolution=1),
                WaveDiscriminator(resolution=2),
                WaveDiscriminator(resolution=4)
            ])
            self.stft_discriminator = STFTDiscriminator()
        
        if args.losses.reconstruction:
            self.rec_loss = ReconstructionLoss()
            # self.rec_loss = ReconstructionLoss2(args.sample_rate)
        
        # VALIDATION OUTPUTS
        self.si_sdr = SISDR()
        self.si_snr = SISNR()
        self.validation_step_outputs = self.reset_valid_outputs()


    def reset_valid_outputs(self):
        return {"pesq": [], 
                "snr": [], 
                "si_sdr": [],
                # "kd_loss": [],
                "input":[], 
                "output":[]}
    
    # OPTIMIZERS
    def configure_optimizers(self):
        lr = self.args.optimizers.lr
        b1 = self.args.optimizers.b1
        b2 = self.args.optimizers.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.model.parameters(),
            ),
            lr=lr, betas=(b1, b2))
        
        if self.args.losses.discriminators:
            optimizer_d = torch.optim.Adam(
                chain(
                    self.wave_discriminators.parameters(),
                    self.stft_discriminator.parameters()
                ),
                lr=lr, betas=(b1, b2))
            return [optimizer_g, optimizer_d], []
        
        return [optimizer_g], []
        
    
    def generate(self, audio_input):
        with torch.no_grad():
            audio_output = self.model(audio_input, "end-to-end")

        return audio_output

    def forward(self, audio_input):
        
        audio_output = self.model(audio_input, "end-to-end")

        return audio_output
    
    # TRAINING
    def train_generator(self, input, output):
        stft_out = self.stft_discriminator(output)
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        self.log("g_stft_loss", g_stft_loss)

        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(3):
            feats1 = self.wave_discriminators[i](input)
            feats2 = self.wave_discriminators[i](output)
            assert len(feats1) == len(feats2)
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(torch.mean(
                torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])) / (len(feats1) - 1)
        self.log("g_wave_loss", g_wave_loss / 3)
        self.log("g_feat_loss", g_feat_loss / 3)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        self.log("g_rec_loss", g_rec_loss)

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss + 100 * g_feat_loss + g_rec_loss
        self.log("g_loss", g_loss, prog_bar=True)
        return g_loss
    
    def train_discriminator(self, input, output):
        stft_out = self.stft_discriminator(input)
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        stft_out = self.stft_discriminator(output)
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        d_wave_loss = 0
        for i in range(3):
            feats = self.wave_discriminators[i](input)
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            feats = self.wave_discriminators[i](output)
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        d_loss = (d_stft_loss + d_wave_loss) / 4

        self.log("d_stft_loss", d_stft_loss)
        self.log("d_wave_loss", d_wave_loss / 3)

        d_loss = (d_stft_loss + d_wave_loss) / 4
        self.log("d_loss", d_loss, prog_bar=True)

        return d_loss
    
    def training_step(self, batch, batch_idx):
        
        # GAN TRAINING:
        if self.args.losses.discriminators:
            optimizer_g, optimizer_d = self.optimizers()

            ## GENERATOR STEP 
            self.toggle_optimizer(optimizer_g)
            
            ### ENCODER STEP
            audio_output = self(batch)
            
            g_loss = self.train_generator(batch, audio_output)

            # SI LOSSES
            if self.args.losses.sisdr:
                sdr_loss = self.si_sdr(batch, audio_output)
                self.log("sdr_loss", sdr_loss)
            else:
                sdr_loss = 0

            if self.args.losses.sisnr:
                snr_loss = self.si_snr(batch, audio_output)
                self.log("snr_loss", snr_loss)
            else:
                snr_loss = 0
            # TOTAL LOSS
            loss = g_loss - sdr_loss - snr_loss #- e_xvector_loss

            self.log("train/encoder_loss", loss)

            self.manual_backward(loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            
            self.untoggle_optimizer(optimizer_g)

            ## DISCRIMINATOR STEP
            self.toggle_optimizer(optimizer_d)
            audio_output = self(batch)
            d_loss = self.train_discriminator(batch, audio_output)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
        
        # STANDARD TRAINING
        else:
            optimizer_g = self.optimizers()

            self.toggle_optimizer(optimizer_g)
            
            audio_output = self(batch)
            
            # REC LOSS
            if self.args.losses.reconstruction:
                g_rec_loss = self.rec_loss(audio_output[:, 0, :], batch[:, 0, :])
                self.log("g_rec_loss", g_rec_loss)
            else:
                g_rec_loss = 0

            # SI LOSSES
            if self.args.losses.sisdr:
                sdr_loss = self.si_sdr(batch, audio_output)
                self.log("sdr_loss", sdr_loss)
            else:
                sdr_loss = 0

            
            if self.args.losses.sisnr:
                snr_loss = self.si_snr(batch, audio_output)
                self.log("snr_loss", snr_loss)
            else:
                snr_loss = 0
            # TOTAL LOSS
            loss = g_rec_loss - sdr_loss - snr_loss

            self.log("train/encoder_loss", loss, prog_bar=True)

            self.manual_backward(loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            
            self.untoggle_optimizer(optimizer_g)

    # VALIDATION
    def validation_step(self, batch, batch_idx ):
        out = self(batch)

        # PESQ
        pesq_score = 0
        for ref, deg in zip(batch, out):
            pesq_score += pesq(16000, ref.squeeze().cpu().numpy(), deg.squeeze().cpu().numpy(), "wb", on_error=1)
        pesq_score /= batch.shape[0]
        self.validation_step_outputs["pesq"].append(pesq_score)

        # SIMILARITY LOSS
        # similarity = self.x_vector_loss(batch, out)
        # self.validation_step_outputs["x_vector_loss"].append(similarity)

        # DISTILL LOSS
        self.validation_step_outputs['si_sdr'].append(self.si_sdr(batch,out))
        self.validation_step_outputs['snr'].append(self.si_snr(batch,out))
        #AUDIO
        self.validation_step_outputs["input"].append(batch)
        self.validation_step_outputs["output"].append(out)

    def on_validation_epoch_end(self):
        if self.logger == "wandb":
            audio_in =  self.validation_step_outputs["input"][0][0]
            audio_out =  self.validation_step_outputs["output"][0][0]
            self.logger.experiment.log({"Input Waveform": wandb.Audio(audio_in.squeeze().cpu().numpy(), sample_rate=16000)})
            self.logger.experiment.log({"Output Waveform": wandb.Audio(audio_out.squeeze().cpu().numpy(), sample_rate=16000)})

        # self.log("val/x_vector_loss", torch.Tensor(self.validation_step_outputs["x_vector_loss"]).mean())
        # self.log("val/kd_loss", torch.Tensor(self.validation_step_outputs["kd_loss"]).mean())
        pesq = self.validation_step_outputs["pesq"]
        self.log("val/pesq", torch.Tensor(pesq).mean())
        
        si_sdr = self.validation_step_outputs["si_sdr"]
        self.log("val/si_sdr", torch.Tensor(si_sdr).mean())
        snr = self.validation_step_outputs["snr"]
        self.log("val/snr", torch.Tensor(snr).mean())

        self.validation_step_outputs = self.reset_valid_outputs()

    # DATASET
    def train_dataloader(self):
        return self._make_dataloader(True)

    def val_dataloader(self):
        return self._make_dataloader(False)
        # return []

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            stacked = torch.stack(examples)

            return stacked.unsqueeze(1)

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, sample_rate, *_ = self._dataset[index]
                x = torchaudio.functional.resample(x, sample_rate, self._sample_rate)
                assert x.shape[0] == 1
                x = torch.squeeze(x)
                x *= 0.95 / torch.max(x)
                assert x.dim() == 1
                if x.shape[0] < self._segment_length:
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                return x

            def __len__(self):
                return len(self._dataset)
                # return 4

        if train:
            ds = torchaudio.datasets.LIBRITTS(self.dataset_args.base_path, url=self.dataset_args.train, download=self.dataset_args.download)
        else:
            ds = torchaudio.datasets.LIBRITTS(self.dataset_args.base_path, url=self.dataset_args.test, download=self.dataset_args.download)

        ds = VoiceDataset(ds, self.args.sample_rate, self.dataset_args.segment_length)
        
        
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.args.batch_size, shuffle=train,
            collate_fn=collate, num_workers=20, pin_memory=True, persistent_workers=True)
        return loader
    
    ### CALLBACKS
    def configure_callbacks(self):
        pass

@hydra.main(version_base=None, config_path='config', config_name='base')
def train(args):

    artifact_url = args.wandb.artifact_url

    #ddp = DDPStrategy( find_unused_parameters=True)
    if args.logger=="wandb":
        logger = WandbLogger(log_model="all", project='soundphi', name="train_01")
    else:
        logger = CSVLogger("logs", name="exp_1")

    if artifact_url != None:
        wandb.init(project="soundphi")  
        # e.g. artifact_dir = 'soundphi/model-8kq0vn4z:v21'
        artifact = wandb.use_artifact(artifact_url, type='model') 
        artifact_dir = artifact.download()  
        model = ExperimentPhi.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    else:
        model = ExperimentPhi(args=args)
        
    trainer = Trainer(logger=logger,
                      devices=args.trainer.devices,
                      #strategy=ddp,
                      accelerator=args.trainer.accelerator,
                      max_steps=args.trainer.max_steps)

    

    trainer.fit(model)


if __name__=="__main__":
    train()
