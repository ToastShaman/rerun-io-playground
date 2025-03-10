import argparse
import rerun as rr
import rerun.blueprint as rrb


def main():
    parser = argparse.ArgumentParser(description="Retail Analytics Demo")

    rr.script_add_args(parser)

    args = parser.parse_args()

    rr.init("retail-analytics-demo", spawn=True)

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


if __name__ == "__main__":
    main()
