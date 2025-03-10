rerun:
    pdm run main_rerun.py

video_processing:
    pdm run video_processing.py

video_capture *ARGS:
    pdm run video_capture.py -- {{ARGS}}

audio_processing:
    pdm run audio_processing.py

audio_capture *ARGS:
    pdm run audio_capture.py -- {{ARGS}}

fmt:
    pdm run ruff format *.py

prepare_offline_models:
    curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt
    (cd ./models && ./download-ggml-model.sh medium.en)
