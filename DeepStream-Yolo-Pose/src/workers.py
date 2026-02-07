"""Background worker management for DB and uploader threads."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
from typing import Iterable, List

from .outputs import OutputManager
from .uploader import Uploader

DB_QUEUE_SIZE = 50000


def create_db_queue(maxsize: int = DB_QUEUE_SIZE) -> Queue:
    """Create the bounded queue used to buffer database events."""
    return Queue(maxsize=maxsize)


def start_db_worker(queue: Queue, outputs: OutputManager) -> Thread:
    """Launch the background thread that forwards queue entries to Event Hub."""
    def _worker() -> None:
        """Consume events from the queue and hand them to the outputs layer."""
        while True:
            try:
                item = queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                (
                    frame_num,
                    video_id,
                    normal_detections,
                    pose_detections,
                    timestamp,
                    camera_id,
                    track_map,
                ) = item
                outputs.process_db_event(
                    (
                        frame_num,
                        video_id,
                        normal_detections,
                        pose_detections,
                        timestamp,
                        camera_id,
                    ),
                    track_map,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"DB worker error: {exc}")
            finally:
                try:
                    queue.task_done()
                except ValueError:
                    pass

    thread = Thread(target=_worker, daemon=True, name="DBWorker")
    thread.start()
    return thread


def start_uploader_workers(camera_ids: Iterable[str]) -> List[Thread]:
    """Spawn one uploader thread per camera to handle cloud syncing."""
    threads: List[Thread] = []
    for cam_id in camera_ids:
        thread = Thread(
            target=_azure_worker,
            args=(cam_id,),
            daemon=True,
            name=f"UploaderWorker-{cam_id}",
        )
        thread.start()
        threads.append(thread)
    return threads


def _azure_worker(camera_id: str) -> None:
    """Run the uploader loop for a single camera id."""
    uploader = Uploader(camera_id=camera_id)
    uploader.loadProcess()
