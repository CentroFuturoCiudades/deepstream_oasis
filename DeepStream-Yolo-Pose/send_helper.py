"""
send_helper.py - Helper class to send inference data to Event Hub

This module provides a SendHelper class with methods to send messages
to different tables (frame, person_observed, detection) via Azure Event Hub.
The messages are consumed by writer.py which inserts them into PostgreSQL.

OPTIMIZED VERSION: Uses async batching with background thread for high throughput.
"""

import os
import json
import uuid
import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Any
from queue import Queue, Empty

from dotenv import load_dotenv
from azure.eventhub import EventHubProducerClient, EventData


class SendHelper:
    """
    Helper class to send inference data to Azure Event Hub with batching.
    
    Messages are buffered and sent in batches to improve throughput.
    Uses a background thread to send events asynchronously.
    
    Supported tables:
        - frame: Video frame metadata
        - person_observed: Tracked person information
        - detection: Individual detection with bounding box and skeleton
    """
    
    def __init__(
        self,
        env_file: Optional[str] = None,
        batch_size: int = 100,
        flush_interval: float = 0.5,
        max_queue_size: int = 50000,
    ):
        """
        Initialize the SendHelper with Event Hub connection.
        
        Args:
            env_file: Path to .env file. If None, uses default .env
            batch_size: Number of events to batch before sending
            flush_interval: Max seconds to wait before flushing partial batch
            max_queue_size: Maximum queue size before blocking
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        self.namespace = os.getenv("EH_NAMESPACE")
        self.eventhub = os.getenv("EH_EVENTHUB")
        self.policy = os.getenv("EH_SEND_POLICY")
        self.key = os.getenv("EH_SEND_KEY")
        
        self.project = os.getenv("PROJECT", "lupacity")
        self.site_id = os.getenv("SITE_ID", "site-01")
        
        self._connection_str = (
            f"Endpoint=sb://{self.namespace}.servicebus.windows.net/;"
            f"SharedAccessKeyName={self.policy};"
            f"SharedAccessKey={self.key};"
            f"EntityPath={self.eventhub}"
        )
        
        # Batching configuration
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        
        # Internal queue for async sending
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None
        self._producer: Optional[EventHubProducerClient] = None
        
        # Statistics
        self._events_queued = 0
        self._events_sent = 0
        self._batches_sent = 0
        self._errors = 0
        self._lock = threading.Lock()
        
        # Start background sender thread
        self._start_sender_thread()
    
    def _start_sender_thread(self):
        """Start the background sender thread."""
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="EventHubSender"
        )
        self._sender_thread.start()
    
    def _get_producer(self) -> EventHubProducerClient:
        """Get or create Event Hub producer."""
        if self._producer is None:
            self._producer = EventHubProducerClient.from_connection_string(
                self._connection_str
            )
        return self._producer
    
    def _sender_loop(self):
        """Background thread that batches and sends events grouped by partition_key."""
        buffer: List[Dict[str, Any]] = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Try to get an event with timeout
                try:
                    event = self._queue.get(timeout=0.1)
                    buffer.append(event)
                    self._queue.task_done()
                except Empty:
                    pass
                
                # Check if we should flush
                should_flush = (
                    len(buffer) >= self._batch_size or
                    (buffer and time.time() - last_flush >= self._flush_interval)
                )
                
                if should_flush and buffer:
                    self._send_batch_by_partition(buffer)
                    buffer = []
                    last_flush = time.time()
                    
            except Exception as e:
                print(f"ERROR in sender loop: {e}")
                with self._lock:
                    self._errors += 1
                time.sleep(0.1)
        
        # Flush remaining events on shutdown
        if buffer:
            try:
                self._send_batch_by_partition(buffer)
            except Exception as e:
                print(f"ERROR flushing final batch: {e}")
    
    def _send_batch_by_partition(self, events: List[Dict[str, Any]]):
        """
        Group events by partition_key and send each group as a separate batch.
        This ensures events with the same partition_key maintain FIFO order.
        """
        if not events:
            return
        
        # Group events by partition_key (defaultdict imported at top)
        partitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for event in events:
            # Extract partition_key from event, default to "default"
            pk = event.pop("_partition_key", "default")
            partitions[pk].append(event)
        
        # Send each partition's events as a batch (maintains order within partition)
        for partition_key, partition_events in partitions.items():
            self._send_batch_internal(partition_events, partition_key)
    
    def _send_batch_internal(self, events: List[Dict[str, Any]], partition_key: Optional[str] = None):
        """Actually send a batch of events to Event Hub with partition_key for ordering."""
        if not events:
            return
            
        try:
            producer = self._get_producer()
            # Create batch with partition_key to ensure FIFO ordering
            batch = producer.create_batch(partition_key=partition_key)
            events_in_batch = 0
            
            for event in events:
                try:
                    batch.add(EventData(json.dumps(event)))
                    events_in_batch += 1
                except ValueError:
                    # Batch is full, send it and create a new one
                    if events_in_batch > 0:
                        producer.send_batch(batch)
                        with self._lock:
                            self._events_sent += events_in_batch
                            self._batches_sent += 1
                    batch = producer.create_batch(partition_key=partition_key)
                    batch.add(EventData(json.dumps(event)))
                    events_in_batch = 1
            
            # Send remaining events in batch
            if events_in_batch > 0:
                producer.send_batch(batch)
                with self._lock:
                    self._events_sent += events_in_batch
                    self._batches_sent += 1
                    
        except Exception as e:
            print(f"ERROR sending batch ({len(events)} events, pk={partition_key}): {e}")
            with self._lock:
                self._errors += 1
    
    def close(self):
        """
        Close the Event Hub producer connection and stop sender thread.
        
        Waits indefinitely for all pending events to be sent.
        Only call this on Ctrl+C / SIGTERM / docker stop.
        """
        pending = self._queue.qsize()
        if pending > 0:
            print(f"Closing SendHelper: waiting for {pending} pending events...")
        
        # Wait for queue to drain completely (no timeout)
        while not self._queue.empty():
            time.sleep(0.1)
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for sender thread to finish (it will flush any remaining buffer)
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join()  # No timeout - wait indefinitely
        
        # Close producer
        if self._producer is not None:
            try:
                self._producer.close()
            except Exception:
                pass
            self._producer = None
        
        # Print stats
        with self._lock:
            print(f"SendHelper closed: {self._events_sent} events sent in {self._batches_sent} batches ({self._errors} errors)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # -------------------------------------------------------------------------
    # Core queue method
    # -------------------------------------------------------------------------
    def _enqueue(self, payload: Dict[str, Any], partition_key: Optional[str] = None):
        """Add an event to the send queue (non-blocking) with partition_key for ordering."""
        # Store partition_key in payload for later extraction
        if partition_key:
            payload["_partition_key"] = partition_key
        
        try:
            self._queue.put_nowait(payload)
            with self._lock:
                self._events_queued += 1
        except Exception:
            # Queue is full, block until space is available (no timeout)
            self._queue.put(payload)  # Blocks indefinitely
            with self._lock:
                self._events_queued += 1
    
    # -------------------------------------------------------------------------
    # Table-specific methods (now queue-based for async sending)
    # -------------------------------------------------------------------------
    def send_frame(
        self,
        frame_id: str,
        camera_id: str,
        timestamp: Optional[str] = None,
        image_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a frame record to Event Hub (queued for async sending)."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        
        payload = {
            "table": "frame",
            "id": frame_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_path": image_path,
            "width": width,
            "height": height,
        }
        
        self._enqueue(payload)
        return payload
    
    def send_person_observed(
        self,
        person_id: str,
        age_group: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a person_observed record to Event Hub (queued for async sending)."""
        payload = {
            "table": "person_observed",
            "id": person_id,
            "age_group": age_group,
        }
        
        self._enqueue(payload)
        return payload
    
    def send_detection(
        self,
        detection_id: str,
        frame_id: str,
        person_id: str,
        confidence: float,
        bbox: Optional[Dict[str, float]] = None,
        skeleton: Optional[List[Dict[str, Any]]] = None,
        px_geometry: Optional[str] = None,
        real_geometry: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a detection record to Event Hub (queued for async sending)."""
        payload = {
            "table": "detection",
            "id": detection_id,
            "frame_id": frame_id,
            "person_id": person_id,
            "confidence": confidence,
            "bbox": bbox,
            "skeleton": skeleton,
            "px_geometry": px_geometry,
            "real_geometry": real_geometry,
        }
        
        self._enqueue(payload)
        return payload
    
    # -------------------------------------------------------------------------
    # Bulk method for maximum efficiency
    # -------------------------------------------------------------------------
    def send_frame_with_detections(
        self,
        frame_id: str,
        camera_id: str,
        timestamp: str,
        width: int,
        height: int,
        detections: List[Dict[str, Any]],
        track_to_person_id: Dict[int, str],
    ) -> int:
        """
        Send a frame and all its detections in one call.
        
        This is the most efficient method for sending detection data.
        All events are queued together and will be batched by the sender thread.
        
        Uses frame_id as partition_key to allow parallel processing across frames,
        while maintaining order within each frame (frame -> person -> detection).
        
        IMPORTANT: person_observed is sent with EVERY detection (idempotent).
        This ensures person_observed always arrives with/before its detection,
        regardless of partition ordering. The writer uses ON CONFLICT DO NOTHING.
        
        Args:
            frame_id: Unique frame identifier (also used as partition_key)
            camera_id: Camera identifier
            timestamp: ISO format timestamp
            width: Frame width
            height: Frame height
            detections: List of detection dicts with keys:
                - track_id: int
                - confidence: float
                - bbox: dict with x, y, width, height
                - skeleton: optional list of keypoints
            track_to_person_id: Mutable dict mapping track_id -> person_id
                (will be updated with new tracks)
        
        Returns:
            Number of events queued
        """
        events_queued = 0
        
        # Use frame_id as partition_key for parallelism across frames
        # All events for this frame go to the same partition (maintains order)
        partition_key = frame_id
        
        # Queue frame first
        self._enqueue({
            "table": "frame",
            "id": frame_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_path": None,
            "width": width,
            "height": height,
        }, partition_key=partition_key)
        events_queued += 1
        
        # Process each detection
        for det in detections:
            track_id = det["track_id"]
            
            # Get or create person_id (track_to_person_id used for consistency)
            if track_id not in track_to_person_id:
                person_id = str(uuid.uuid4())
                track_to_person_id[track_id] = person_id
            else:
                person_id = track_to_person_id[track_id]
            
            # ALWAYS send person_observed (idempotent - ON CONFLICT DO NOTHING)
            # This guarantees person exists before detection, regardless of partition order
            self._enqueue({
                "table": "person_observed",
                "id": person_id,
                "age_group": None,
            }, partition_key=partition_key)
            events_queued += 1
            
            # Queue detection
            self._enqueue({
                "table": "detection",
                "id": str(uuid.uuid4()),
                "frame_id": frame_id,
                "person_id": person_id,
                "confidence": det.get("confidence", 0.0),
                "bbox": det.get("bbox"),
                "skeleton": det.get("skeleton"),
                "px_geometry": None,
                "real_geometry": None,
            }, partition_key=partition_key)
            events_queued += 1
        
        return events_queued
    
    # -------------------------------------------------------------------------
    # Stats and control
    # -------------------------------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        """Get sending statistics."""
        with self._lock:
            return {
                "events_queued": self._events_queued,
                "events_sent": self._events_sent,
                "batches_sent": self._batches_sent,
                "errors": self._errors,
                "queue_size": self._queue.qsize(),
            }
    
    def flush(self):
        """
        Wait for all queued events to be sent.
        
        Blocks indefinitely until queue is empty.
        """
        initial_size = self._queue.qsize()
        
        if initial_size > 0:
            print(f"Flushing {initial_size} pending events...")
        
        while not self._queue.empty():
            time.sleep(0.1)
    
    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"[SendHelper] Queued: {stats['events_queued']} | "
              f"Sent: {stats['events_sent']} | "
              f"Batches: {stats['batches_sent']} | "
              f"Queue: {stats['queue_size']} | "
              f"Errors: {stats['errors']}")


# -----------------------------------------------------------------------------
# Utility functions for standalone usage
# -----------------------------------------------------------------------------
_default_helper: Optional[SendHelper] = None


def get_default_helper() -> SendHelper:
    """Get or create a default SendHelper instance."""
    global _default_helper
    if _default_helper is None:
        _default_helper = SendHelper()
    return _default_helper


def close_default_helper():
    """Close the default helper if it exists."""
    global _default_helper
    if _default_helper is not None:
        _default_helper.close()
        _default_helper = None


# -----------------------------------------------------------------------------
# Example usage / test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print(" SendHelper Batching Test")
    print("=" * 50)
    
    with SendHelper(batch_size=50, flush_interval=0.5) as helper:
        camera_id = "-1"
        track_to_person = {}
        
        print("\n📤 Sending 100 frames with 5 detections each...")
        start = time.time()
        
        for frame_num in range(100):
            frame_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            detections = [
                {
                    "track_id": i,
                    "confidence": 0.9,
                    "bbox": {"x": 100 + i*50, "y": 100, "width": 50, "height": 120},
                    "skeleton": None,
                }
                for i in range(5)
            ]
            
            helper.send_frame_with_detections(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp=timestamp,
                width=1920,
                height=1080,
                detections=detections,
                track_to_person_id=track_to_person,
            )
        
        # Wait for all events to be sent
        print("⏳ Flushing...")
        helper.flush()
        
        elapsed = time.time() - start
        stats = helper.get_stats()
        
        print(f"\n✅ Done in {elapsed:.2f}s")
        print(f"   Events queued: {stats['events_queued']}")
        print(f"   Events sent: {stats['events_sent']}")
        print(f"   Batches sent: {stats['batches_sent']}")
        if elapsed > 0:
            print(f"   Throughput: {stats['events_sent']/elapsed:.0f} events/sec")
    
    print("\n" + "=" * 50)
