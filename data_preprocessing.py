import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sample = "0/1.wav"
data, sample_rate = librosa.load(sample)

plt.title("Wave form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

all_data = []

data_path_dict = {
    0: ["0/" + file_name for file_name in os.listdir("0/")],
    1: ["1/" + file_name for file_name in os.listdir("1/")]
}

for class_label, list_of_files in data_path_dict.items():
    for file in list_of_files:
        data, sample_rate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfccs.T, axis=0)
        all_data.append([mfcc_processed, class_label])
    print(f"Successfully preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])
df.to_pickle("final_data/final_data.csv")