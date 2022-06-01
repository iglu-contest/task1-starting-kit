from azure.storage.blob import ContainerClient
import os
from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dest', type=str, required=True)
args = parser.parse_args()

class BlobFileDownloader:
    def __init__(self, local_blob_path, sas_token):
        # self.sas_token = os.getenv("IGLU_Data_Collection_Blob_SAS")
        self.sas_token = sas_token
        self.sas_url = "https://igludatacollection.blob.core.windows.net/iglu-data-task-1?" + self.sas_token
        self.container_client = ContainerClient.from_container_url(self.sas_url)
        self.local_blob_path = local_blob_path

    def list_blobs(self):
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print(blob.name + '\n')
    def __save_blob__(self,file_name,file_content):
        # Get full path to the file
        download_file_path = os.path.join(self.local_blob_path, file_name)
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        
        with open(download_file_path, "wb") as file:
            file.write(file_content)
    
    def upload_files_to_blob(self, SOURCE_FILE, blob_name):
        with open(SOURCE_FILE, "rb") as data:
            blob_client = self.container_client.upload_blob(name=blob_name, data=data)

    def download_blobs_in_container(self):
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print(blob.name + '\n')
            content = self.container_client.get_blob_client(blob).download_blob().readall()
            self.__save_blob__(blob.name, content)

if __name__ == "__main__":
    sas_token = 'sp=rl&st=2022-05-14T00:40:31Z&se=2022-12-31T08:40:31Z&spr=https&sv=2020-08-04&sr=c&sig=Fp1H0DI164%2BKP8wTRQsZbaQTsPUdwNnZE7VV7xKGMxc%3D'
    # If you are using Azure explorer, you can access the container via url: 
    # 'https://igludatacollection.blob.core.windows.net/iglu-data-task-1?sp=rl&st=2021-10-26T00:20:33Z&se=2021-12-31T09:20:33Z&spr=https&sv=2020-08-04&sr=c&sig=lBNi7Qgl7KWCSY%2FW0Loua3tdjKjnaDCOvJxn39sPT1Q%3D'
    local_blob_path = args.save_dest
    blob_obj = BlobFileDownloader(local_blob_path, sas_token)
    blob_obj.list_blobs()
    blob_obj.download_blobs_in_container()
