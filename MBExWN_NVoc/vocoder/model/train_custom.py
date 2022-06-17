import os
import sys

import tensorflow as tf
import tensorflow_tts.configs.melgan as MELGAN_CONFIG
from MBExWN_NVoc import get_config_file
from MBExWN_NVoc.vocoder.dataset.pickle_mel_dataset import PickleMelDataset
from MBExWN_NVoc.vocoder.model import config_utils as cutils
from MBExWN_NVoc.vocoder.model.models import create_model
from tensorflow_tts.models import TFMelGANMultiScaleDiscriminator
from tensorflow_tts.utils import calculate_3d_loss


class MBExWN_trainer():
    def __init__(self, model_id_or_path: str, verbose=True) -> None:
        self.verbose = verbose
        self.model_dir, self.hparams = self._get_model_dir_and_hparams(
            model_id_or_path)
        self.generator, self.hparams = self._load_model(
            self.model_dir, self.hparams)
        # TODO: Add discriminator weights loading too
        self.discriminator = TFMelGANMultiScaleDiscriminator(
            MELGAN_CONFIG.MelGANDiscriminatorConfig(
                **self.hparams["melgan_discriminator_params"]
            ),
            name="melgan_discriminator",
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _get_model_dir_and_hparams(self, model_id_or_path):
        config_file = get_config_file(model_id_or_path=model_id_or_path)
        model_dir = os.path.dirname(config_file)
        hparams = cutils.read_config(config_file=config_file)
        return model_dir, hparams

    def _load_model(self, model_dir, hparams):
        training_config = hparams["training_config"]
        self.preprocess_config = hparams["preprocess_config"]

        # ## Instantiate model and optimizer
        model, mr_mode = create_model(
            hparams,
            training_config,
            self.preprocess_config,
            quiet=True,
        )

        # we need to run the model at least once si that all components are built otherwise the
        # state that is loaded from the checkpoint will disappear once the model is run
        # the first time when all layers are built.
        # Configure for arbitrary sound sizes
        model.build_model(variable_time_dim=True)

        model_weights_path = os.path.join(model_dir, "weights.tf")
        if tf.compat.v1.train.checkpoint_exists(model_weights_path):
            if self.verbose:
                print(f"restore from {model_weights_path}", file=sys.stderr)
            model.load_weights(model_weights_path)
        return model, hparams

    def adv_loss(self, disc_outs):
        adv_loss = 0.0
        for i in range(len(disc_outs)):
            adv_loss += calculate_3d_loss(
                tf.ones_like(disc_outs[i][-1]), disc_outs[i][-1], loss_fn=self.mse_loss
            )
        adv_loss /= i + 1
        adv_loss = tf.reduce_mean(adv_loss)
        return adv_loss

    def lr_loss(self, ins, outs, step=0):
        total_loss, spect_loss, mel_loss, NPOW_loss, *block_losses = self.generator.total_loss(
            outs, ins, step)
        return spect_loss
    
    def f0_loss(self):
        return self.generator.block.F0_loss

    def forward_pass(self, x):
        gen = self.generator(x, training=True)
        disc = self.discriminator(tf.expand_dims(gen, -1))
        return gen, disc

    def train(self, root_dir, batch_size):
        ds = PickleMelDataset(root_dir)
        ds = iter(ds.create(batch_size=batch_size))
        while True:
            batch = next(ds)
            audios = tf.expand_dims(batch["audios"], -1)
            mels = batch["mels"]
            # Placeholder until f0s are correctly loaded
            f0 = tf.random.uniform([audios.shape[0], 1, 1])
            gen, disc = self.forward_pass(
                [audios, mels, f0])
            adv_loss = self.adv_loss(disc)
            lr_loss = self.lr_loss(audios, gen)
            f0_loss = self.f0_loss()
            print(
                f"Gen shape: {gen.shape}\nDisc len: {len(disc)}\nAdv loss: {adv_loss}\nLR loss: {lr_loss}\nF0 loss: {f0_loss}")


if __name__ == "__main__":
    trainer = MBExWN_trainer("SPEECH")
    root_dir = "/tmp/VCTK_subset_2000_mels/"
    trainer.train(root_dir, 2)
