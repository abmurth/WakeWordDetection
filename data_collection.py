import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path, n_times=100):
    input("Click Enter to start recording")
    for i in range(n_times):
        sample_rate= 44100
        seconds = 2
        recording = sd.rec(int(seconds*sample_rate), samplerate=sample_rate, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", sample_rate, recording)
        input(f"Click enter to record next sample or Ctrl + C to exit {i+1}/{n_times}")

def record_background_save(save_path, n_times = 99):
    input("Click Enter to start recording")
    for i in range(n_times):
        sample_rate= 44100
        seconds = 2
        recording = sd.rec(int(seconds*sample_rate), samplerate=sample_rate, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", sample_rate, recording)
        print(f"Recording {i+1}/{n_times}")
# print("Recording the wake word: \n")
# record_audio_and_save("1/")

print("Recording background sound")
record_background_save("0/")
