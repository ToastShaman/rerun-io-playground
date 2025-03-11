import argparse
import os
from dotenv import load_dotenv
import rerun as rr
import rerun.blueprint as rrb


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Retail Analytics Demo")

    rr.script_add_args(parser)

    args = parser.parse_args()

    recording_id = os.getenv("RECORDING_ID")

    rr.init("retail-analytics-demo", recording_id=recording_id, spawn=True)

    rr.script_setup(
        args,
        "retail-analytics-demo",
        default_blueprint=rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/image/original", name="Video"),
                rrb.Spatial2DView(origin="/image/yolo", name="Processed Video"),
            ),
            rrb.TextDocumentView(origin="/text/transcript", name="Transcript"),
        ),
    )

    rr.script_teardown(args)
