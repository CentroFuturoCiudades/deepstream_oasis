"""Entry point for DeepStream YOLO Pose application."""

from __future__ import annotations
import sys
import time
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from src import classifier_worker, config, outputs, pipeline, workers


def main() -> int:
    """Launch the DeepStream pipelines and manage their lifecycle."""
    app_config = config.load_app_config()

    Gst.init(None)
    loop = GLib.MainLoop()

    runtime = config.create_runtime_config(app_config, pipeline.is_jetson())
    output_manager = outputs.OutputManager(app_config, runtime)
    output_manager.init_csv()
    output_manager.init_event_hub()

    classification_worker_instance: (
        classifier_worker.TrackletClassificationWorker | None
    )
    classification_worker_instance = None
    classification_worker_instance = classifier_worker.TrackletClassificationWorker(
        camera_ids=app_config.arguments.camera_ids
    )
    classification_worker_instance.start()

    db_queue = workers.create_db_queue()
    workers.start_db_worker(db_queue, output_manager)

    pipelines: list[pipeline.CameraPipeline] = []
    for idx, (source_uri, camera_id) in enumerate(
        zip(app_config.arguments.sources, app_config.arguments.camera_ids)
    ):
        camera_pipeline = pipeline.CameraPipeline(
            index=idx,
            source_uri=source_uri,
            camera_id=camera_id,
            app_config=app_config,
            runtime=runtime,
            outputs=output_manager,
            db_queue=db_queue,
            loop=loop,
        )
        if camera_pipeline.error:
            print(f"ERROR: {camera_pipeline.error}")
            output_manager.close_csv()
            output_manager.cleanup_event_hub()
            return 1
        pipelines.append(camera_pipeline)

    workers.start_uploader_workers(app_config.arguments.camera_ids)

    summary = [
        f"SOURCES:   {', '.join(app_config.arguments.sources)}",
        f"CONFIG:    {app_config.arguments.infer_config}",
        f"OUTPUT:    {app_config.arguments.output_path}",
        f"SIZE:      {runtime.width}x{runtime.height}",
        f"GPU:       {runtime.gpu_id}",
        f"JETSON:    {runtime.is_jetson}",
        *output_manager.summary_lines(),
    ]

    print(f"\n{'=' * 50}")
    for line in summary:
        print(line)
    if len(pipelines) == 1:
        print(f"CAMERA_ID: {pipelines[0].camera_id}")
    else:
        joined = ", ".join(str(cam.camera_id) for cam in pipelines)
        print(f"CAMERA_IDS: {joined}")
    print(f"{'=' * 50}\n")

    try:
        while True:
            print("Starting pipelines...")
            for pipe in pipelines:
                pipe.start()
            loop.run()
            print("Pipelines stopped, restarting in 3s...")
            for pipe in pipelines:
                pipe.stop()
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        for pipe in pipelines:
            pipe.stop()

    if classification_worker_instance:
        classification_worker_instance.stop()

    output_manager.close_csv()

    print(f"\nOutput saved: {app_config.arguments.output_path}")
    if runtime.enable_csv:
        print(f"Metadata saved: {app_config.constants.csv_path}")
    if runtime.enable_db:
        print("Database records sent via Event Hub")
        total_persons = sum(len(cam.track_to_person_id) for cam in pipelines)
        print(f"Total tracked persons: {total_persons}")

    print("Waiting for DB queue to flush...")
    db_queue.join()
    print("All DB events sent")

    output_manager.cleanup_event_hub()
    return 0


if __name__ == "__main__":
    sys.exit(main())
