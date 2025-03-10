import pyaudio
import wave
import asyncio
import os
from datetime import datetime

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 25
INPUT_DEVICE_INDEX = 1  # Adjust for your mic
WHISPER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "ggml-medium.en.bin"
)

AUDIO_DIR = os.path.join(os.path.dirname(__file__), ".audio")

# Create the output directory if it doesn't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Async queue for transcription tasks
transcription_queue = asyncio.Queue()


def record_audio(loop):
    """Continuously records 25-second chunks and adds them to the queue."""
    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=INPUT_DEVICE_INDEX,
            frames_per_buffer=CHUNK,
        )

        print("Recording started. Press Ctrl+C to stop.")

        while True:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"audio_chunk_{timestamp}.wav"
            output_filename_path = os.path.join(
                os.path.dirname(__file__), ".audio", output_filename
            )

            print(f"Recording {output_filename} for {RECORD_SECONDS} seconds...")
            frames = [
                stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))
            ]

            # Save as WAV file
            with wave.open(output_filename_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))

            print(f"Audio saved as {output_filename}")

            # Add file to async queue (ensuring it's done inside the event loop)
            asyncio.run_coroutine_threadsafe(
                transcription_queue.put(output_filename_path), loop
            )

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def run_whisper_cli():
    """Process audio files from the queue using Whisper CLI and delete them after transcription."""
    while True:
        filename = await transcription_queue.get()
        print(f"Transcribing {filename}...")

        whisper_cmd = [
            "whisper-cli",
            "-m",
            WHISPER_MODEL_PATH,
            "-f",
            filename,
            "--output-json",
        ]

        process = await asyncio.create_subprocess_exec(
            *whisper_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if stdout:
            print(f"Transcription output:\n{stdout.decode()}")
        if stderr:
            print(f"Errors:\n{stderr.decode()}")

        # Delete file after processing
        os.remove(filename)
        print(f"Deleted {filename}")

        transcription_queue.task_done()


async def main():
    """Start the recording and transcription processes."""
    loop = asyncio.get_running_loop()  # Get current event loop

    # Run recording in a separate thread, passing the event loop
    asyncio.get_event_loop().run_in_executor(None, record_audio, loop)

    # Run transcription in async
    await run_whisper_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
