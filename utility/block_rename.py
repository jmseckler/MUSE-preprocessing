import os
import pandas as pd

def rename_zarr_file(old_filepath, excel_filepath):

    """Converts sample block ID to REVA approved block ID

    Parameters
    ----------
    old_filepath : str
        The Directory housing the stitched acquisition run .zarr files
    excel_filepath : str
        Excel file containing two columns 'Block ID' and 'Reva ID'

    Returns
    -------
    new_filepath: str
        The converted filepath name of the concatenated acquisition run .zarr files
    """

    # Extract the basename
    sample_block = os.path.basename(os.path.normpath(old_filepath))
    
    print("Sample Block:", sample_block)

    # Extract the relevant part of the basename (CL1-1)
    block_id_part = '-'.join(sample_block.split('-')[-2:])
    
    print("Block ID Part:", block_id_part)

    # Read the Excel file
    df = pd.read_excel(excel_filepath)
    print("DataFrame:\n", df)

    # Find the corresponding REVA_ID
    new_id_row = df[df['Block_ID'] == block_id_part]
    
    if not new_id_row.empty:
        new_id = new_id_row['REVA_ID'].values[0]
        print("New ID:", new_id)
        
        # Construct the new filepath
        new_filepath = os.path.join(os.path.dirname(old_filepath), new_id)
        return new_filepath
    else:
        print("No matching Block ID found.")
        return None

# old_filepath = '/mnt/smb-share/processed-data/SR005/SR005-review/SR005-CL1-1'
# excel_filepath = '/home/nmj14/Documents/REVA/block_name_convert.xlsx'

# new_filepath = rename_zarr_file(old_filepath, excel_filepath)
# print("New Filepath:", new_filepath)


