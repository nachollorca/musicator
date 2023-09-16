from pydub import AudioSegment
import pandas as pd
import librosa
import numpy as np
import joblib


class Song:
    def __init__(self, file):
        self.file = file
        self.scaler = joblib.load("checkpoints/scaler.pkl")
        self.working_file = "tmp/working_file.wav"
        self.features = None
        self.standardized_features = None

    def load(self) -> None:
        filetype = self.file.name[-3:]
        print(filetype)
        audio = AudioSegment.from_file(self.file, format="wav")

        # Get middle minute if it is longer than 60 seconds
        if len(audio) > 60000:
            print("Cropping middle segment")
            start_time = len(audio) // 2 - 30000
            end_time = len(audio) // 2 + 30000
            audio = audio[start_time:end_time]

        audio.export(self.working_file, format="wav")
        pass

    def extract_features(self) -> pd.DataFrame:
        try:
            x, sr = librosa.load(self.working_file)
        except Exception as e:
            print(f"An error occured while loading the audio as a floating point time series: \n"
                  f"{str(e)} \n"
                  f"Did you use `Song.load()` already to generate the working file?")
            return None

        # Calculate tempo
        tempo, _ = librosa.beat.beat_track(y=x, sr=sr)

        # Calculate MFCC features
        mfcc = librosa.feature.mfcc(y=x, sr=sr)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        # Calculate other audio features
        chroma = librosa.feature.chroma_stft(y=x, sr=sr)
        rms = librosa.feature.rms(y=x)
        centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y=x)
        harmonic = librosa.effects.harmonic(y=x)
        percussive = librosa.effects.percussive(y=x)

        # Order as in the original df
        features = {
            "length": 66149,
            "chroma_stft_mean": np.mean(chroma),
            "chroma_stft_var": np.var(chroma),
            "rms_mean": np.mean(rms),
            "rms_var": np.var(rms),
            "spectral_centroid_mean": np.mean(centroid),
            "spectral_centroid_var": np.var(centroid),
            "spectral_bandwidth_mean": np.mean(bandwidth),
            "spectral_bandwidth_var": np.var(bandwidth),
            "rolloff_mean": np.mean(rolloff),
            "rolloff_var": np.var(rolloff),
            "zero_crossing_rate_mean": np.mean(zero_crossing),
            "zero_crossing_rate_var": np.var(zero_crossing),
            "harmony_mean": np.mean(harmonic),
            "harmony_var": np.var(harmonic),
            "perceptr_mean": np.mean(percussive),
            "perceptr_var": np.var(percussive),
            "tempo": tempo,
            "mfcc1_mean": mfcc_mean[0],
            "mfcc1_var": mfcc_var[0],
            "mfcc2_mean": mfcc_mean[1],
            "mfcc2_var": mfcc_var[1],
            "mfcc3_mean": mfcc_mean[2],
            "mfcc3_var": mfcc_var[2],
            "mfcc4_mean": mfcc_mean[3],
            "mfcc4_var": mfcc_var[3],
            "mfcc5_mean": mfcc_mean[4],
            "mfcc5_var": mfcc_var[4],
            "mfcc6_mean": mfcc_mean[5],
            "mfcc6_var": mfcc_var[5],
            "mfcc7_mean": mfcc_mean[6],
            "mfcc7_var": mfcc_var[6],
            "mfcc8_mean": mfcc_mean[7],
            "mfcc8_var": mfcc_var[7],
            "mfcc9_mean": mfcc_mean[8],
            "mfcc9_var": mfcc_var[8],
            "mfcc10_mean": mfcc_mean[9],
            "mfcc10_var": mfcc_var[9],
            "mfcc11_mean": mfcc_mean[10],
            "mfcc11_var": mfcc_var[10],
            "mfcc12_mean": mfcc_mean[11],
            "mfcc12_var": mfcc_var[11],
            "mfcc13_mean": mfcc_mean[12],
            "mfcc13_var": mfcc_var[12],
            "mfcc14_mean": mfcc_mean[13],
            "mfcc14_var": mfcc_var[13],
            "mfcc15_mean": mfcc_mean[14],
            "mfcc15_var": mfcc_var[14],
            "mfcc16_mean": mfcc_mean[15],
            "mfcc16_var": mfcc_var[15],
            "mfcc17_mean": mfcc_mean[16],
            "mfcc17_var": mfcc_var[16],
            "mfcc18_mean": mfcc_mean[17],
            "mfcc18_var": mfcc_var[17],
            "mfcc19_mean": mfcc_mean[18],
            "mfcc19_var": mfcc_var[18],
            "mfcc20_mean": mfcc_mean[19],
            "mfcc20_var": mfcc_var[19],
        }

        features = pd.DataFrame(features, index=[0])
        self.features = features
        return features

    def standardize_features(self):
        try:
            standardized_features = self.scaler.transform(self.features)
        except Exception as e:
            print(f"An error occured while standardizing the features: \n"
                  f"{str(e)} \n"
                  f"Did you use `Song.extract_features()` already to populate the song features?")
            return None

        standardized_features = pd.DataFrame(standardized_features, columns=self.features.columns)
        self.standardized_features = standardized_features
        return standardized_features