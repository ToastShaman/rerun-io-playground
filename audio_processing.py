from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import os
import subprocess
import torch
import zmq
import argparse
from dotenv import load_dotenv


AUDIO_DIR = os.path.join(os.path.dirname(__file__), ".audio")

WHISPER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "ggml-medium.en.bin"
)

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def transcribe(**kwargs):
    try:
        timestamp = datetime.datetime.now().isoformat()
        filename = f"{timestamp}_audio_chunk.wav"
        output_filename = os.path.join(AUDIO_DIR, filename)

        # Write the binary audio data to a file
        binary_audio_data = kwargs["binary_audio_data"]
        with open(output_filename, "wb") as f:
            f.write(binary_audio_data)

        # Add the filename and timestamp to the kwargs
        args = {**kwargs, "timestamp": timestamp, "filename": output_filename}

        # Transcribe the audio using both models in parallel
        with ProcessPoolExecutor() as executor:
            future1 = executor.submit(transcribe_with_whisper, **args)
            future2 = executor.submit(transcribe_with_pyannote, **args)

            # Wait for both futures to complete
            transcriptions = future1.result()
            diarization = future2.result()

        return transcriptions, diarization
    finally:
        os.remove(output_filename)


def transcribe_with_pyannote(**kwargs):
    """
    Transcribe the audio file using the Pyannote speaker diarization model
    """

    from pyannote.audio import Pipeline

    filename, access_token = kwargs["filename"], kwargs["access_token"]

    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=access_token)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    pipeline.to(device)

    diarization = pipeline(filename)

    return diarization


def transcribe_with_whisper(**kwargs):
    """
    Transcribe the audio file using the Whisper model
    """

    filename, timestamp = kwargs["filename"], kwargs["timestamp"]

    try:
        whisper_cmd = [
            "whisper-cli",
            "-m",
            WHISPER_MODEL_PATH,
            "-f",
            filename,
            "--output-json",
        ]

        # Run the whisper-cli command
        subprocess.run(
            whisper_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Read the JSON output
        json_output_filename = f"{filename}.json"
        with open(json_output_filename, "r") as f:
            json_data = json.load(f)

        # Add the timestamp to the JSON data
        json_data["timestamp"] = timestamp

        return json_data
    finally:
        os.remove(json_output_filename)


def main(**kwargs):
    zmq_address, access_token = (
        kwargs["zmq_address"],
        kwargs["access_token"],
    )

    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(zmq_address)

        print(f"Listening for audio data on {zmq_address}")

        while True:
            binary_audio_data = socket.recv()

            # Transcribe the audio data
            transcriptions, diarization = transcribe(
                binary_audio_data=binary_audio_data, access_token=access_token
            )

            # TODO: Implement speaker segmentation and merging

            socket.send_json(transcriptions)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing server")
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="tcp://*:6666",
        help="ZeroMQ server address to bind to (e.g., tcp://*:6666)",
    )

    args = parser.parse_args()

    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)

    load_dotenv()

    main(
        zmq_address=args.zmq_address,
        access_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    )
