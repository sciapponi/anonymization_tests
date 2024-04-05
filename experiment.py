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

class Experiment(L.LightningModule):

    def __init__(self,
                 lr: float = 0.0002,
                 b2: float = 0.999,):
        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Speaker Embeddings
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
    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        ### ENCODER STEP
        self.decoder.requires_grad = False 
        audio_output, quantized = self(batch)

        e_distill_loss = self.distillation_loss(quantized, batch)
        
        e_xvector_loss = self.x_vector_loss(batch, audio_output)
        
        loss = e_distill_loss - e_xvector_loss

        self.log("train/encoder_distill_loss", e_distill_loss)
        self.log("train/encoder_xvector_loss", e_xvector_loss)
        self.log("train/encoder_loss", loss)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        ### DECODER STEP
        self.decoder.requires_grad = True
        self.encoder.requires_grad = False 
        self.quantizer.requires_grad = False 
        audio_output, quantized = self(batch)

        d_distill_loss = self.distillation_loss(quantized, batch)

        d_xvector_loss = self.x_vector_loss(batch, audio_output)

        loss = d_distill_loss - d_xvector_loss

        self.log("train/decoder_distill_loss", d_distill_loss)
        self.log("train/decoder_xvector_loss", d_xvector_loss)
        self.log("train/decoder_loss", loss)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

    def on_validation_epoch_end(self):
        waveform = [[1]]
        self.logger.experiment.log({"Audio": wandb.Audio(waveform[0][0], sample_rate=16000)})

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
        

    def configure_callbacks(self):
        pass

if __name__=="__main__":
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(logger=wandb_logger)

    train_set = datasets.LIBRITTS(root=".", url="train-clean-100", download=True)
    test_set = datasets.LIBRITTS(root=".", url="test-clean", download=True)
    model = Experiment()
    trainer.fit(model)
