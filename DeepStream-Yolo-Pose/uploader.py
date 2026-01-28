"""
uploader.py - Async frame uploader to Azure Blob Storage

Monitors a local directory for JPEG files and uploads them to Azure Blob Storage.
Files are deleted after successful upload.

Frame path format in Azure: {camera_id}/{timestamp}.jpg
Local path format: /tmp/frames/{timestamp}.jpg
"""

import os
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict
from pathlib import Path

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("WARNING: azure.storage.blob not available. Frame uploads disabled.")

from dotenv import load_dotenv


# Default directory for frame files
DEFAULT_FRAME_DIR = "/tmp/frames"


class FrameUploader:
    """
    Async frame uploader to Azure Blob Storage.
    
    Monitors a local directory for JPEG files and uploads them to Azure.
    Files are deleted after successful upload.
    """
    
    def __init__(
        self,
        env_file: Optional[str] = None,
        container_name: str = "oasis-ds",
        frame_dir: str = DEFAULT_FRAME_DIR,
        camera_id: str = "-1",
        num_workers: int = 2,
        poll_interval: float = 0.1,
    ):
        """
        Initialize the FrameUploader.
        
        Args:
            env_file: Path to .env file. If None, uses default .env
            container_name: Azure Blob container name
            frame_dir: Local directory to monitor for frame files
            camera_id: Camera identifier for blob path prefix
            num_workers: Number of upload worker threads
            poll_interval: How often to check for new files (seconds)
        """
        if not AZURE_AVAILABLE:
            raise RuntimeError("azure-storage-blob not available")
        
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        account_url = os.getenv("AZURE_ACCOUNT_URL")
        sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
        
        if not account_url or not sas_token:
            raise ValueError("Azure credentials not found: AZURE_ACCOUNT_URL, AZURE_STORAGE_SAS_TOKEN")
        
        self._container_name = container_name
        self._frame_dir = Path(frame_dir)
        self._camera_id = camera_id
        self._poll_interval = poll_interval
        
        # Create frame directory if it doesn't exist
        self._frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Azure client
        self._blob_service = BlobServiceClient(account_url, credential=sas_token)
        self._container_client = self._blob_service.get_container_client(container_name)
        
        # Queue for files to upload
        self._queue: Queue = Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._workers: list = []
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._frames_queued = 0
        self._frames_uploaded = 0
        self._errors = 0
        self._lock = threading.Lock()
        
        # Start workers and monitor
        self._num_workers = num_workers
        self._start_workers()
        self._start_monitor()
    
    def _start_workers(self):
        """Start the background upload worker threads."""
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._upload_worker,
                daemon=True,
                name=f"FrameUploader-{i}"
            )
            worker.start()
            self._workers.append(worker)
    
    def _start_monitor(self):
        """Start the directory monitor thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_directory,
            daemon=True,
            name="FrameUploader-Monitor"
        )
        self._monitor_thread.start()
    
    def _monitor_directory(self):
        """Monitor directory for new JPEG files and queue them for upload."""
        seen_files = set()
        
        while not self._stop_event.is_set():
            try:
                # Get all .jpg files in directory
                current_files = set(self._frame_dir.glob("*.jpg"))
                
                # Find new files
                new_files = current_files - seen_files
                
                for file_path in new_files:
                    # Wait briefly to ensure file is fully written
                    time.sleep(0.01)
                    
                    try:
                        # Queue for upload
                        self._queue.put_nowait(str(file_path))
                        with self._lock:
                            self._frames_queued += 1
                    except Exception:
                        pass  # Queue full, will retry next cycle
                
                # Update seen files (only keep existing ones)
                seen_files = current_files
                
            except Exception as e:
                print(f"ERROR in directory monitor: {e}")
            
            time.sleep(self._poll_interval)
    
    def _upload_worker(self):
        """Background worker that uploads files from queue."""
        content_settings = ContentSettings(content_type="image/jpeg")
        
        while not self._stop_event.is_set():
            try:
                # Get file path from queue with timeout
                try:
                    file_path = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                
                try:
                    # Extract timestamp from filename (filename format: {timestamp}.jpg)
                    file_path = Path(file_path)
                    timestamp = file_path.stem  # Gets filename without extension
                    
                    # Generate blob path: camera_id/timestamp.jpg
                    blob_path = f"{self._camera_id}/{timestamp}.jpg"
                    
                    # Read file content
                    with open(file_path, "rb") as f:
                        jpeg_bytes = f.read()
                    
                    # Upload to Azure Blob
                    blob_client = self._container_client.get_blob_client(blob_path)
                    blob_client.upload_blob(
                        jpeg_bytes,
                        overwrite=True,
                        content_settings=content_settings
                    )
                    
                    # Delete local file after successful upload
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # File may have been deleted already
                    
                    with self._lock:
                        self._frames_uploaded += 1
                        
                except FileNotFoundError:
                    # File was deleted before we could upload
                    pass
                except Exception as e:
                    print(f"ERROR uploading {file_path}: {e}")
                    with self._lock:
                        self._errors += 1
                finally:
                    self._queue.task_done()
                    
            except Exception as e:
                print(f"ERROR in upload worker: {e}")
                time.sleep(0.1)
    
    def queue_file(self, file_path: str) -> bool:
        """
        Manually queue a file for upload (alternative to auto-monitoring).
        
        Args:
            file_path: Path to the JPEG file to upload
            
        Returns:
            True if queued successfully, False if queue is full
        """
        try:
            self._queue.put_nowait(file_path)
            with self._lock:
                self._frames_queued += 1
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict:
        """
        Get upload statistics.
        
        Returns:
            Dictionary with queued, uploaded, errors counts
        """
        with self._lock:
            return {
                "frames_queued": self._frames_queued,
                "frames_uploaded": self._frames_uploaded,
                "errors": self._errors,
                "queue_size": self._queue.qsize(),
            }
    
    def flush(self, timeout: float = 30.0) -> bool:
        """
        Wait for all queued uploads to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if queue was flushed, False if timeout
        """
        try:
            start = time.time()
            while self._queue.qsize() > 0:
                if time.time() - start > timeout:
                    return False
                time.sleep(0.1)
            return True
        except Exception:
            return False
    
    def close(self):
        """Stop all workers and cleanup."""
        self._stop_event.set()
        
        # Wait for workers to finish current uploads
        for worker in self._workers:
            worker.join(timeout=2.0)
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)


def get_frame_directory() -> str:
    """
    Get the frame directory path, creating it if necessary.
    
    Returns:
        Path to frame directory
    """
    frame_dir = Path(DEFAULT_FRAME_DIR)
    frame_dir.mkdir(parents=True, exist_ok=True)
    return str(frame_dir)
