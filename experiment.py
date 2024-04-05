import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import soundstream
import torch 
from torch import nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from itertools import chain
import wandb
from torchaudio import datasets
from discriminators import WaveDiscriminator, STFTDiscriminator
from losses import ReconstructionLoss

class Experiment(L.LightningModule):

    def __init__(self, 
                    lr: float = 1e-4,
                    b1: float = 0.5,
                    b2: float = 0.9,):
        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # X-VECTORS
        self.speaker_embedder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        self.speaker_embedder.requires_grad = False

        # SOUNDSTREAM
        audio_codec = soundstream.from_pretrained()

        self.encoder = audio_codec['encoder']
        self.quantizer = audio_codec['quantizer']
        self.decoder = audio_codec['decoder']

        # HUBERT
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
        self.hubert.requires_grad = False

        # DISCRIMINATORS
        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()

    # OPTIMIZERS
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        return [optimizer_g, optimizer_d], []
    
    def forward(self, audio_input):
        #SOUNDSTREAM
        x = self.encoder(audio_input)
        quantized, _, _ = self.quantizer(x.permute(0,2,1))
        audio_output = self.decoder(quantized.permute(0,2,1))

        return audio_output, quantized
    
    # LOSSES
    def distillation_loss(self, z, audio_input):

        # Distillation loss: makes the SoundStream (quantized) Embedding space have hubert-like soft-speech rapresentations.

        hubert_features = self.hubert(audio_input)
        loss_distill = (z - F.interpolate(hubert_features, z.shape[2])).abs().mean()

        return loss_distill
    
    def x_vector_loss(self, y, y_hat):

        # X vector loss: xvectors from the input and output audio should be as dissimilar as possible:
        # We achieve this maximizing the CosineSimilarity between the two speaker embeddings.

        y_xvector =  self.speaker_embedder.encode_batch(y)
        y_hat_xvector = self.speaker_embedder.encode_batch(y)

        return nn.CosineSimilarity(y_xvector, y_hat_xvector)

    def compute_loss(self, quantized, y, y_hat):
        
        return self.distillation_loss(quantized, y) - self.x_vector_loss(y, y_hat)
    
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
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss + 100 * g_feat_loss + g_rec_loss

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
        self.decoder.requires_grad = False 
        audio_output, quantized = self(batch)
        
        g_loss = self.train_generator(batch, audio_output)

        e_distill_loss = self.distillation_loss(quantized, batch)
        
        e_xvector_loss = self.x_vector_loss(batch, audio_output)
        
        loss = g_loss + e_distill_loss - e_xvector_loss

        self.log("train/encoder_distill_loss", e_distill_loss)
        self.log("train/encoder_xvector_loss", e_xvector_loss)
        self.log("train/encoder_loss", loss)

        self.manual_backward(loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        ### DECODER STEP
        self.decoder.requires_grad = True
        self.encoder.requires_grad = False 
        self.quantizer.requires_grad = False 
        audio_output, quantized = self(batch)

        g_loss = self.train_generator(batch, audio_output)

        d_distill_loss = self.distillation_loss(quantized, batch)

        d_xvector_loss = self.x_vector_loss(batch, audio_output)

        loss = g_loss + d_distill_loss - d_xvector_loss

        self.log("train/decoder_distill_loss", d_distill_loss)
        self.log("train/decoder_xvector_loss", d_xvector_loss)
        self.log("train/decoder_loss", loss)

        self.manual_backward(loss)
        optimizer_g.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        ## DISCRIMINATOR STEP
        self.toggle_optimizer(optimizer_d)
        audio_output, quantized = self(batch)
        d_loss = self.train_discriminator(batch, audio_output)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)



    def on_validation_epoch_end(self):
        waveform = [[1]]
        self.logger.experiment.log({"Audio": wandb.Audio(waveform[0][0], sample_rate=16000)})
        
    
        

    def configure_callbacks(self):
        pass

if __name__=="__main__":
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(logger=wandb_logger)

    train_set = datasets.LIBRITTS(root=".", url="train-clean-100", download=True)
    test_set = datasets.LIBRITTS(root=".", url="test-clean", download=True)
    model = Experiment()
    trainer.fit(model)
