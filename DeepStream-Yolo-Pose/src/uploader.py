from azure.storage.blob import BlobServiceClient
import os
from glob import glob
from dotenv import load_dotenv
import subprocess
# local imports 
import time

class Uploader():

    def __init__(self, camera_id) -> None:
        """
        Class builder, initializes the Uploader with account URL and SAS token.
        
        Args: 
            account_url: str
            sas_token: str
        Returns:
            [None]: None
        """
        super().__init__()
        load_dotenv()  # Load environment variables from .env file
        account_url = os.getenv("AZURE_ACCOUNT_URL")
        sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
        
        if not account_url or not sas_token:
            raise ValueError("Azure credentials not found in environment variables")

        self.camera_id = camera_id
        print('Initializing Uploader from .env')
        self.blob_service_client = BlobServiceClient(account_url, credential=sas_token)
        print('Uploader initialized')

    def upload_video(self, local_file_name: str) -> None:
        """
        Upload a video file to Azure Blob Storage.
        
        Args: 
            local_file_name: str
        Returns:
            [None]: None
        """
        print(f"Uploading video to Azure Storage: {local_file_name}")
        try:
            blob_name = f"{self.camera_id}/{os.path.basename(local_file_name)}"
            blob_client = self.blob_service_client.get_blob_client(container="oasis-ds", blob=blob_name)

            print(f"Uploading to Azure Storage as blob: {local_file_name}")

            with open(file=local_file_name, mode="rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            print(f"Uploaded {local_file_name} to Azure Blob Storage.")

        except Exception as e:
            print(f"Error uploading {local_file_name} to Azure Blob Storage: {e}")


    def loadProcess(self) -> None:
        """
        This method upload all .mp4 files and delete them from output folder.

        Args:
            None
        Returns:
            None
        """
        print('Uploader worker started')

        while(True):
            files = glob(f'output/{self.camera_id}/*.mp4',recursive=True)
            print(f'Loaded files: {files}')
            for f in files:
                filename = os.path.basename(f)

                if filename.startswith("temp"):
                    print(f'Skipping temp file {f}')
                    continue
                
                # Evit files that are being written
                if time.time() - os.path.getmtime(f) < 5:
                    continue

                try:
                    print(f'Uploading {f}')
                    self.upload_video(f)

                    os.remove(f)
                    print(f'Deleted file {f}')

                except Exception as e:
                    print(
                        f'Failed processing {f}: {e}'
                    )
            time.sleep(1)