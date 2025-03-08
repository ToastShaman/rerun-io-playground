rerun:
    pdm run rerun

video_capture *ARGS:
    pdm run video_capture.py -- {{ARGS}}

video_processing:
    pdm run video_processing.py

fmt:
    pdm run ruff format *.py
