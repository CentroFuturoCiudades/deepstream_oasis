"""
send_helper.py - Helper class to send inference data to Event Hub

This module provides a SendHelper class with methods to send messages
to different tables (frame, person_observed, detection) via Azure Event Hub.
The messages are consumed by writer.py which inserts them into PostgreSQL.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from azure.eventhub import EventHubProducerClient, EventData


class SendHelper:
    """
    Helper class to send inference data to Azure Event Hub.
    
    Messages are structured with a 'table' field that indicates
    the destination PostgreSQL table in writer.py.
    
    Supported tables:
        - frame: Video frame metadata
        - person_observed: Tracked person information
        - detection: Individual detection with bounding box and skeleton
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the SendHelper with Event Hub connection.
        
        Args:
            env_file: Path to .env file. If None, uses default .env
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
        
        self._producer: Optional[EventHubProducerClient] = None
    
    @property
    def producer(self) -> EventHubProducerClient:
        """Lazy initialization of Event Hub producer."""
        if self._producer is None:
            self._producer = EventHubProducerClient.from_connection_string(
                self._connection_str
            )
        return self._producer
    
    def close(self):
        """Close the Event Hub producer connection."""
        if self._producer is not None:
            self._producer.close()
            self._producer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # -------------------------------------------------------------------------
    # Core send method
    # -------------------------------------------------------------------------
    def _send_event(self, payload: Dict[str, Any], partition_key: Optional[str] = None):
        """
        Send a single event to Event Hub.
        
        Args:
            payload: Dictionary with data to send (must include 'table' field)
            partition_key: Optional partition key for routing
        """
        batch = self.producer.create_batch(partition_key=partition_key)
        batch.add(EventData(json.dumps(payload)))
        
        if partition_key:
            self.producer.send_batch(batch)
        else:
            self.producer.send_batch(batch)
    
    def _send_batch(self, payloads: List[Dict[str, Any]], partition_key: Optional[str] = None):
        """
        Send multiple events to Event Hub in a single batch.
        
        Args:
            payloads: List of dictionaries to send
            partition_key: Optional partition key for routing
        """
        if not payloads:
            return
            
        batch = self.producer.create_batch()
        for payload in payloads:
            batch.add(EventData(json.dumps(payload)))
        
        if partition_key:
            self.producer.send_batch(batch, partition_key=partition_key)
        else:
            self.producer.send_batch(batch)
    
    # -------------------------------------------------------------------------
    # Table-specific methods
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
        """
        Send a frame record to Event Hub.
        
        Args:
            frame_id: Unique identifier for the frame
            camera_id: Camera identifier
            timestamp: ISO format timestamp (defaults to current UTC time)
            image_path: Path to the saved frame image
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            The payload that was sent
        """
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
        
        self._send_event(payload, partition_key=camera_id)
        return payload
    
    def send_person_observed(
        self,
        person_id: str,
        age_group: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a person_observed record to Event Hub.
        
        Args:
            person_id: Unique identifier for the tracked person
            age_group: Age group classification (e.g., 'adult', 'child')
            camera_id: Optional camera ID for partition routing
            
        Returns:
            The payload that was sent
        """
        payload = {
            "table": "person_observed",
            "id": person_id,
            "age_group": age_group,
        }
        
        self._send_event(payload, partition_key=camera_id)
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
        """
        Send a detection record to Event Hub.
        
        Args:
            detection_id: Unique identifier for this detection
            frame_id: Reference to the parent frame
            person_id: Reference to the tracked person
            confidence: Detection confidence score (0.0 - 1.0)
            bbox: Bounding box dict with keys: x, y, width, height
            skeleton: List of keypoints for pose estimation
            px_geometry: WKT geometry string for pixel coordinates
            real_geometry: WKT geometry string for real-world coordinates
            camera_id: Optional camera ID for partition routing
            
        Returns:
            The payload that was sent
        """
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
        
        self._send_event(payload, partition_key=camera_id)
        return payload
    
    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------
    def send_full_detection(
        self,
        camera_id: str,
        person_id: str,
        confidence: float,
        bbox: Optional[Dict[str, float]] = None,
        skeleton: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[str] = None,
        image_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        age_group: Optional[str] = None,
        px_geometry: Optional[str] = None,
        real_geometry: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Send a complete detection with frame and person records.
        
        This convenience method creates and sends all three related records:
        frame, person_observed, and detection.
        
        Args:
            camera_id: Camera identifier
            person_id: Unique identifier for the tracked person
            confidence: Detection confidence score
            bbox: Bounding box dictionary
            skeleton: Pose keypoints list
            timestamp: ISO format timestamp
            image_path: Path to saved frame image
            width: Frame width
            height: Frame height
            age_group: Person age classification
            px_geometry: Pixel coordinate geometry (WKT)
            real_geometry: Real-world coordinate geometry (WKT)
            
        Returns:
            Dictionary with generated IDs: frame_id, person_id, detection_id
        """
        frame_id = str(uuid.uuid4())
        detection_id = str(uuid.uuid4())
        
        # Send all three records
        self.send_frame(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=timestamp,
            image_path=image_path,
            width=width,
            height=height,
        )
        
        self.send_person_observed(
            person_id=person_id,
            age_group=age_group,
            camera_id=camera_id,
        )
        
        self.send_detection(
            detection_id=detection_id,
            frame_id=frame_id,
            person_id=person_id,
            confidence=confidence,
            bbox=bbox,
            skeleton=skeleton,
            px_geometry=px_geometry,
            real_geometry=real_geometry,
            camera_id=camera_id,
        )
        
        return {
            "frame_id": frame_id,
            "person_id": person_id,
            "detection_id": detection_id,
        }
    
    def send_batch_detections(
        self,
        detections: List[Dict[str, Any]],
        camera_id: str,
        frame_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        image_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send multiple detections for a single frame.
        
        Args:
            detections: List of detection dicts, each containing:
                - person_id: str
                - confidence: float
                - bbox: Optional dict
                - skeleton: Optional list
                - age_group: Optional str
                - px_geometry: Optional str (WKT)
                - real_geometry: Optional str (WKT)
            camera_id: Camera identifier
            frame_id: Optional frame ID (generated if not provided)
            timestamp: ISO format timestamp
            image_path: Path to saved frame image
            width: Frame width
            height: Frame height
            
        Returns:
            Dictionary with frame_id and list of detection_ids
        """
        if frame_id is None:
            frame_id = str(uuid.uuid4())
        
        # Send frame first
        self.send_frame(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=timestamp,
            image_path=image_path,
            width=width,
            height=height,
        )
        
        detection_ids = []
        
        # Send each detection
        for det in detections:
            person_id = det.get("person_id", str(uuid.uuid4()))
            detection_id = str(uuid.uuid4())
            
            # Send person if needed
            self.send_person_observed(
                person_id=person_id,
                age_group=det.get("age_group"),
                camera_id=camera_id,
            )
            
            # Send detection
            self.send_detection(
                detection_id=detection_id,
                frame_id=frame_id,
                person_id=person_id,
                confidence=det.get("confidence", 0.0),
                bbox=det.get("bbox"),
                skeleton=det.get("skeleton"),
                px_geometry=det.get("px_geometry"),
                real_geometry=det.get("real_geometry"),
                camera_id=camera_id,
            )
            
            detection_ids.append(detection_id)
        
        return {
            "frame_id": frame_id,
            "detection_ids": detection_ids,
        }


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


def send_frame(**kwargs) -> Dict[str, Any]:
    """Convenience function to send a frame record."""
    return get_default_helper().send_frame(**kwargs)


def send_person_observed(**kwargs) -> Dict[str, Any]:
    """Convenience function to send a person_observed record."""
    return get_default_helper().send_person_observed(**kwargs)


def send_detection(**kwargs) -> Dict[str, Any]:
    """Convenience function to send a detection record."""
    return get_default_helper().send_detection(**kwargs)


def send_full_detection(**kwargs) -> Dict[str, str]:
    """Convenience function to send a complete detection with all records."""
    return get_default_helper().send_full_detection(**kwargs)


# -----------------------------------------------------------------------------
# Example usage / test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print(" SendHelper Test")
    print("=" * 50)
    
    # Using context manager
    with SendHelper() as helper:
        camera_id = "-1"
        
        # Example 1: Send individual records
        print("\n📤 Sending individual records...")
        
        frame_id = str(uuid.uuid4())
        person_id = f"track-{int(time.time())}"
        detection_id = str(uuid.uuid4())
        
        helper.send_frame(
            frame_id=frame_id,
            camera_id=camera_id,
            width=1920,
            height=1080,
        )
        print(f"   ✅ Frame sent: {frame_id[:8]}...")
        
        helper.send_person_observed(
            person_id=person_id,
            age_group="adult",
            camera_id=camera_id,
        )
        print(f"   ✅ Person sent: {person_id}")
        
        helper.send_detection(
            detection_id=detection_id,
            frame_id=frame_id,
            person_id=person_id,
            confidence=0.92,
            bbox={"x": 100, "y": 200, "width": 50, "height": 120},
            skeleton=[
                {"name": "nose", "x": 125, "y": 210, "confidence": 0.95},
                {"name": "left_eye", "x": 120, "y": 205, "confidence": 0.93},
            ],
            camera_id=camera_id,
        )
        print(f"   ✅ Detection sent: {detection_id[:8]}...")
        
        # Example 2: Send full detection (convenience method)
        print("\n📤 Sending full detection...")
        
        ids = helper.send_full_detection(
            camera_id=camera_id,
            person_id=f"track-{int(time.time()) + 1}",
            confidence=0.88,
            bbox={"x": 300, "y": 150, "width": 60, "height": 140},
            width=1920,
            height=1080,
            age_group="adult",
        )
        print(f"   ✅ Full detection sent:")
        print(f"      Frame: {ids['frame_id'][:8]}...")
        print(f"      Person: {ids['person_id']}")
        print(f"      Detection: {ids['detection_id'][:8]}...")
        
        # Example 3: Send batch detections
        print("\n📤 Sending batch detections...")
        
        batch_result = helper.send_batch_detections(
            camera_id=camera_id,
            width=1920,
            height=1080,
            detections=[
                {
                    "person_id": f"track-batch-1",
                    "confidence": 0.91,
                    "bbox": {"x": 100, "y": 100, "width": 50, "height": 120},
                },
                {
                    "person_id": f"track-batch-2",
                    "confidence": 0.85,
                    "bbox": {"x": 400, "y": 150, "width": 55, "height": 130},
                },
            ],
        )
        print(f"   ✅ Batch sent:")
        print(f"      Frame: {batch_result['frame_id'][:8]}...")
        print(f"      Detections: {len(batch_result['detection_ids'])}")
    
    print("\n" + "=" * 50)
    print(" All events sent successfully!")
    print("=" * 50)
