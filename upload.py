import globus_sdk
from globus_sdk.scopes import TransferScopes

#Source and Destination endpoint IDs
SRC = "SOURCE ENDPOINT ID"
DST = "3264b50a-9a71-40fe-b6bd-c8e0729b94f4"

#base path to MUSE folder on globus
DST_PATH = "/ShoffLab/REVA/2AllStagingData/MUSE/source-DONOTUPLOAD/"

#service account information
CLIENT_ID = "d83b29f2-469a-4225-8e73-d72ba38205be"
CLIENT_SECRET = "secret, do not upload to github"

#transfers the file/directory located at path
def transfer_data_to_globus(path):
    conf_client = globus_sdk.ConfidentialAppAuthClient(CLIENT_ID, CLIENT_SECRET)

    #Create transfer client
    scopes = TransferScopes.all
    cc_authorizer = globus_sdk.ClientCredentialsAuthorizer(conf_client, scopes)
    transfer_client = globus_sdk.TransferClient(authorizer=cc_authorizer)
    
    task_data = build_task_data(path)
    
    #submit transfer
    task_doc = transfer_client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    print(f"submitted transfer, task_id={task_id}")
    t = 0
    while not transfer_client.task_wait(task_id, timeout=60):
        t+=1
        print(f"transfer in progress, {t} minutes")
    
    successful_transfers = transfer_client.task_successful_transfers("18cc2828-00e3-11ef-932f-7b1e52fc573b")["DATA"]
    paths = [t["source_path"] for t in successful_transfers]
    print("Files transferred successfully: " + paths)

#create task data object for file/directory located at path
def build_task_data(path):
    #Generate task data for transfer task
    task_data = globus_sdk.TransferData(
        source_endpoint=SRC, destination_endpoint=DST, verify_checksum=True
        )

    task_data.add_item(
        path,
        DST_PATH + path
        )
    
    return task_data





