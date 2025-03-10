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
brew install just pyenv whisper-cpp
brew reinstall portaudio
```

## Setup

### Prepare Offline Models

Download and prepare the necessary machine learning models:

```bash
just prepare_offline_models
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