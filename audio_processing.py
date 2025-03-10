import datetime
import json
import os
import subprocess
import zmq
import argparse

AUDIO_DIR = os.path.join(os.path.dirname(__file__), ".audio")

WHISPER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "ggml-medium.en.bin"
)


def transcribe(binary_audio_data):
    timestamp = datetime.datetime.now().isoformat()
    filename = f"{timestamp}_audio_chunk.wav"
    output_filename = os.path.join(AUDIO_DIR, filename)

    with open(output_filename, "wb") as f:
        f.write(binary_audio_data)

    return transcribe_with_whisper(timestamp, output_filename)


def transcribe_with_whisper(timestamp, filename):
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
        os.remove(filename)
        os.remove(json_output_filename)


def main(zmq_address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(zmq_address)

        print(f"Listening for audio data on {zmq_address}")

        while True:
            binary_audio_data = socket.recv()
            transcription = transcribe(binary_audio_data)
            socket.send_json(transcription)
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

    main(args.zmq_address)
