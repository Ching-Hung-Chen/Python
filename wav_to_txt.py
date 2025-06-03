import wave
import numpy as np
import matplotlib.pyplot as plt

# ---------- 使用者參數設定 ----------
filename = r"C:\Users\joeych\Desktop\voice_1281475.wav"     # .wav 檔案名稱
plot_seconds = 3           # 只畫前幾秒的音訊
threshold = 10000             # 振幅跳變大於此值視為異常點
# -----------------------------------

# 讀取 .wav 檔
with wave.open(filename, 'rb') as wav_file:
    n_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()

    # 計算要讀取的 frame 數
    max_samples = int(frame_rate * plot_seconds)
    max_samples = min(max_samples, n_frames)  # 避免超出檔案長度

    raw_data = wav_file.readframes(max_samples)

# 轉換為 numpy array
if sample_width == 2:
    audio_data = np.frombuffer(raw_data, dtype=np.int16)
elif sample_width == 1:
    audio_data = np.frombuffer(raw_data, dtype=np.uint8)
else:
    raise ValueError("Unsupported sample width")

# 解析聲道資料
if n_channels == 2:
    audio_data = np.reshape(audio_data, (-1, 2))
    left_channel = audio_data[:, 0]
else:
    left_channel = audio_data

# 建立時間軸
time_axis = np.linspace(0, len(left_channel) / frame_rate, num=len(left_channel))

# 偵測異常點（振幅突變大於門檻）
diff = np.abs(np.diff(left_channel))
anomaly_indices = np.where(diff > threshold)[0]
anomaly_times = time_axis[anomaly_indices]

# 畫出波形 + 高亮異常點
plt.figure(figsize=(12, 4))
plt.plot(time_axis, left_channel, label="Waveform")
plt.scatter(anomaly_times, left_channel[anomaly_indices], color='red', s=15, label='Anomaly')
plt.title(f"Waveform (First {plot_seconds} sec) with Anomaly Highlight")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
