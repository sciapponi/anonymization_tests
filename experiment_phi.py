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
from losses import ReconstructionLoss#, XVectorLoss
# from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from pesq import pesq
from model import SoundPhi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio import  SignalNoiseRatio as SNR

torch.set_float32_matmul_precision('medium')

# if torch.cuda.is_available(): 
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device('cuda')

class ExperimentPhi(L.LightningModule):

    def __init__(self, 
                 use_pretrained = False,
                 batch_size:int = 16,
                 sample_rate: int = 16000,
                 segment_length: int = 48000,
                 latent_space_dim = 64,
                 lr: float = 1e-4,
                 b1: float = 0.5,
                 b2: float = 0.9,):
        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = SoundPhi(latent_space_dim=latent_space_dim,
                              n_q=16,
                              codebook_size=1024)
        

        # DISCRIMINATORS
        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()

        # self.x_vector_loss = XVectorLoss()
        #PESQ

        # VALIDATION OUTPUTS
        self.si_sdr = SISDR()
        self.snr = SNR()
        self.validation_step_outputs = self.reset_valid_outputs()

        # self.ce_loss = nn.CrossEntropyLoss()

    def reset_valid_outputs(self):
        return {"pesq": [], 
                "snr": [], 
                "si_sdr": [],
                # "kd_loss": [],
                "input":[], 
                "output":[]}
    
    # OPTIMIZERS
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.model.parameters(),
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        
        return [optimizer_g, optimizer_d], []
    
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

        optimizer_g, optimizer_d = self.optimizers()

        ## GENERATOR STEP 
        self.toggle_optimizer(optimizer_g)
        
        ### ENCODER STEP
        audio_output = self(batch)
        
        g_loss = self.train_generator(batch, audio_output)

        #e_xvector_loss = self.x_vector_loss(batch, audio_output)
        
        loss = g_loss #- e_xvector_loss

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
        self.validation_step_outputs['snr'].append(self.snr(batch,out))
        #AUDIO
        self.validation_step_outputs["input"].append(batch)
        self.validation_step_outputs["output"].append(out)

    def on_validation_epoch_end(self):
        audio_in =  self.validation_step_outputs["input"][0][0]
        audio_out =  self.validation_step_outputs["output"][0][0]
        # self.logger.experiment.log({"Input Waveform": wandb.Audio(audio_in.squeeze().cpu().numpy(), sample_rate=16000)})
        # self.logger.experiment.log({"Output Waveform": wandb.Audio(audio_out.squeeze().cpu().numpy(), sample_rate=16000)})

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
                # return 100

        if train:
            ds = torchaudio.datasets.LIBRITTS("/home/ste/Datasets/", url="train-clean-360", download=True)
        else:
            ds = torchaudio.datasets.LIBRITTS("/home/ste/Datasets/", url="test-clean", download=True)

        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)
        
        
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams.batch_size, shuffle=train,
            collate_fn=collate, num_workers=20, pin_memory=True, persistent_workers=True)
        return loader
    
    ### CALLBACKS
    def configure_callbacks(self):
        pass


def train():
    #ddp = DDPStrategy( find_unused_parameters=True)
    logger = WandbLogger(log_model="all", project='soundphi', name="train_01")
    # logger = CSVLogger("logs", name="exp_1")
    trainer = Trainer(logger=logger,
                      devices=1,
                      #strategy=ddp,
                      accelerator='gpu',
                      max_steps=1000000)

    model = ExperimentPhi(batch_size=16,)
    # trainer.fit(model)
    trainer.fit(model)


if __name__=="__main__":
    train()
