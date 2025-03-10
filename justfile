rerun:
    pdm run rerun

video_capture *ARGS:
    pdm run video_capture.py -- {{ARGS}}

video_processing:
    pdm run video_processing.py

fmt:
    pdm run ruff format *.py

install:
    curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt
    (cd ./models && ./download-ggml-model.sh medium.en)
