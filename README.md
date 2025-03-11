# Retail Analytics Demo

This project demonstrates a retail analytics system that captures video and audio data, processes it using machine learning models, and provides insights in real-time.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Homebrew](https://brew.sh/)
- [Just](https://github.com/casey/just)
- [Pyenv](https://github.com/pyenv/pyenv)
- [Whisper-CPP](https://github.com/ggerganov/whisper.cpp)
- [PortAudio](http://www.portaudio.com/)

You can install the required dependencies using the following commands:

```bash
brew install just pyenv cmake automake whisper-cpp
brew reinstall portaudio
```

## Setup

### Prepare Offline Models

Download and prepare the necessary machine learning models:

```bash
just prepare_offline_models
```

### Download Pre-trained Model for Speaker Diarization

To download the pre-trained model for speaker diarization, follow these steps:

1. **Go to Hugging Face**: Visit the [Hugging Face](https://huggingface.co/).
1. **Create an Account**: If you don't have an account, create one and log in.
1. **Go to Pyannote Speaker Diarization Model**: Visit the [Pyannote Diarization Hugging Face model page](https://huggingface.co/pyannote/speaker-diarization-3.1).
1. **Agree**: Provide your company name and website. Click 'Agree and access repository'
1. **Go to Pyannote Segmentation Model**: Visit the [Pyannote Segmentation Hugging Face model page](https://huggingface.co/pyannote/segmentation-3.0).
1. **Agree**: Provide your company name and website. Click 'Agree and access repository'
1. **Generate an Access Token**:
   - Navigate to your [Hugging Face settings](https://huggingface.co/settings/tokens).
   - Click on **New token**.
   - Name your token and select the appropriate scopes (e.g., `read`).
   - Click **Generate** and copy the token.
1. **Create a `.env` File**:
   - In the root directory of your project, create a file named `.env`.
   - Add the following line to the `.env` file, replacing `YOUR_ACCESS_TOKEN` with the token you copied:
   
     ```env
     HUGGINGFACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN
     ```

### Set Up Python Environment

Use `pyenv` to set up the Python environment:

```bash
pyenv install 3.13.2
```   

### Install Python Dependencies

Install the required Python packages:

```bash
pdm install
```

## Running the Demo

To start the video and audio capture and processing:

```bash
just rerun

just video_processing
just video_capture --device 1

just audio_processing
just audio_capture --device 1
```