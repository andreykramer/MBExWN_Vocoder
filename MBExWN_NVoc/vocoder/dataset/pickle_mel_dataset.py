import math
import os
import random

import numpy as np
import tensorflow as tf
from MBExWN_NVoc import get_config_file, mel_inverter
from MBExWN_NVoc.fileio import iovar as iov
from MBExWN_NVoc.sig_proc.resample import resample
from MBExWN_NVoc.vocoder.model import config_utils as cutils
from pysndfile import sndio
from tensorflow_tts.datasets import AbstractDataset
from tensorflow_tts.utils import find_files


class PickleMelDataset(AbstractDataset):
    """Tensorflow compatible mel dataset that loads files generated by generate_mel script."""

    def __init__(
        self,
        root_dir,
        audio_query="*.wav",
        mel_ext="mell",
        example_duration=2,
        model_id_or_path="SPEECH",
    ):
        """Initialize dataset.
        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_ext (str): Extension of feature files.
            example_duration (float): Desired length in seconds of each example.
            model_id_or_path (str): id or path to model.
        """
        # find all of mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # assert the number of files
        assert len(
            audio_files) != 0, f"Not found any mel files in ${root_dir}."

        utt_ids = [PickleMelDataset._get_audio_basename(
            f) for f in audio_files]
        audio_and_mel = [(f, PickleMelDataset._find_mel_of_audio(f, mel_ext))
                         for f in audio_files]

        self.utt_ids = utt_ids
        self.audio_and_mel = audio_and_mel

        # Model parameters
        self.model_id_or_path = model_id_or_path
        self.model_config = self._get_model_config(model_id_or_path)
        preprocess_config = self.model_config["preprocess_config"]
        self.hop_size = preprocess_config["hop_size"]
        self.sr = preprocess_config["sample_rate"]
        self.mel_channels = preprocess_config["mel_channels"]

        # Example parameters
        self.example_duration = example_duration
        mel_frames_per_audio_sec = self._get_frames_per_sec()
        self.frames_per_seg = math.ceil(
            mel_frames_per_audio_sec * self.example_duration)

    @staticmethod
    def _get_audio_basename(audio_path):
        return os.path.splitext(os.path.basename(audio_path))[0]

    @staticmethod
    def _find_mel_of_audio(audio_path, mel_ext):
        root_dir = os.path.dirname(audio_path)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        mel_path = os.path.join(root_dir, f"{basename}.{mel_ext}")
        return mel_path

    def _get_model_config(self, model_id_or_path):
        config_file = get_config_file(model_id_or_path=model_id_or_path)
        hparams = cutils.read_config(config_file=config_file)
        return hparams

    def _get_frames_per_sec(self):
        """ Returns amount of mel frames that match an audio duration of [example_duration]
        for the given model.
        """
        segment_length = self.sr / self.hop_size
        return segment_length

    def _load_audio(self, audio_file):
        snd, sr, _ = sndio.read(audio_file, dtype=np.dtype("float32"))
        if sr != self.sr:
            snd, _ = resample(
                snd, sr, self.sr, axis=0)
        return snd

    def _load_mel(self, path):
        dd = iov.load_var(path)
        MelInv = mel_inverter.MELInverter(
            model_id_or_path=self.model_id_or_path)
        mel = MelInv.scale_mel(dd)[0, ...]
        return mel

    def _load_f0(self, path):
        f0 = iov.load_var(path)["f0_vals"]
        return f0

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            audio_file, mel_file = self.audio_and_mel[i]
            audio = self._load_audio(audio_file)
            mel = self._load_mel(mel_file)
            mel_length = mel.shape[0]
            example_length = self.example_duration * self.sr
            if audio.shape[0] >= example_length:
                # Cut to example length beginning from a random position
                mel_max_start = mel_length - self.frames_per_seg - 1
                mel_start_idx = random.randint(0, mel_max_start)
                mel = mel[mel_start_idx:mel_start_idx+self.frames_per_seg, :]
                audio = audio[mel_start_idx *
                              self.hop_size:(mel_start_idx + self.frames_per_seg) * self.hop_size]
            else:
                # Pad to example length
                mel = tf.pad(
                    mel, [[0, self.segment_lenght - mel_length], [0, 0]], 'constant')
                audio = tf.pad(
                    mel, [[0, example_length - audio.shape[0]], [0, 0]], 'constant')
            # TODO: Replace with correct
            f0 = tf.expand_dims(audio, -1)
            items = {"utt_ids": utt_id, "mels": mel, "audios": audio, "f0": f0}

            yield items

    def get_output_shapes(self):
        output_shapes = {
            "utt_ids": tf.TensorShape([]),
            # TODO parametrize the 80
            "mels": tf.TensorShape([self.frames_per_seg, self.mel_channels]),
            "audios": tf.TensorShape([self.example_duration * self.sr]),
            "f0": tf.TensorShape([self.example_duration * self.sr, 1])
        }
        return output_shapes

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mels": tf.float32,
            "audios": tf.float32,
            "f0": tf.float32
        }
        return output_types

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        output_shapes = self.get_output_shapes()
        print(output_shapes)
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, output_shapes=output_shapes, args=(
                self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        datasets = datasets.batch(batch_size)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "PickleMelDataset"


# Just for testing, will be removed.
if __name__ == "__main__":
    root_dir = "/tmp/VCTK_subset_2000_mels/"
    ds = PickleMelDataset(root_dir)
    print(ds._load_mel("/tmp/VCTK_subset_2000_mels/vctk_281_1241.mell").shape)
    while True:
        print(next(iter(ds.create(batch_size=2))))
