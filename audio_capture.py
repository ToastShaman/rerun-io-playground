import argparse
import io
import wave
import zmq
import pyaudio
import asyncio
import rerun as rr

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10

captured_transcripts = []


def capture_audio(audio_device_index, transcription_queue, loop):
    """
    Capture audio from a microphone and put it into a transcription queue.
    """
    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=audio_device_index,
            frames_per_buffer=CHUNK,
        )

        while True:
            # Record audio
            frames = [
                stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))
            ]

            # Join the frames together
            audio_data = b"".join(frames)

            # Save as WAV file in memory
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)

            # Seek to the beginning of the BytesIO buffer
            audio_buffer.seek(0)

            # Get the binary audio data
            binary_audio_data = audio_buffer.getvalue()

            # Put the audio data into the transcription queue
            asyncio.run_coroutine_threadsafe(
                transcription_queue.put_nowait(binary_audio_data), loop
            )
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def callback_factory(zmq_address, transcription_queue):
    """
    Create a callback function that processes each audio frame.
    """
    context = zmq.Context()
    zmq_socket = context.socket(zmq.REQ)
    zmq_socket.connect(zmq_address)

    async def callback():
        while True:
            # Get the audio data from the transcription queue
            audio_data = await transcription_queue.get()

            # Send the audio data to the ZeroMQ server
            zmq_socket.send(audio_data)

            # Wait for the reply
            results = zmq_socket.recv_json()

            # Log the transcript
            results = " ".join(
                [item["text"] for item in results["transcription"]]
            ).strip()

            # Add the transcript to the list
            captured_transcripts.append(results)

            # Log the transcript
            rr.log(
                "text/transcript",
                rr.TextDocument(
                    "\n\n".join(captured_transcripts), media_type=rr.MediaType.MARKDOWN
                ),
            )

            transcription_queue.task_done()

    return zmq_socket, context, callback


async def main(device, zmq_address):
    rr.init("retail-analytics-demo", spawn=True)

    transcription_queue = asyncio.Queue()

    loop = asyncio.get_event_loop()

    try:
        zmq_socket, context, callback = await callback_factory(
            zmq_address, transcription_queue
        )
        loop.run_in_executor(None, capture_audio, device, transcription_queue, loop)
        await callback()
    finally:
        zmq_socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture audio from a device")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which audio device to use. (Passed to `PyAudio()`)",
    )
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="tcp://localhost:6666",
        help="ZeroMQ server address (e.g., tcp://localhost:6666)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.device, args.zmq_address))
