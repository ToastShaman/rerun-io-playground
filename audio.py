import pyaudio
import wave

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"


# Function to list all audio input devices
def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
    audio.terminate()


# Function to record audio
def record_audio(device_index):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    frames = []

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


# List all audio devices
list_audio_devices()

# Replace 'device_index' with the index of your webcam's microphone
device_index = int(input("Enter the device index of your webcam's microphone: "))

# Start audio recording
record_audio(device_index)
