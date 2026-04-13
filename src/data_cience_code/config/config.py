
DATASET_FOLDER_ABS_PATH = "/home/murlock/Descargas/data-cience-code/src/dataset"
OUTPUT_FOLDER_ABS_PATH = "/home/murlock/Descargas/data-cience-code/src/output"

# Dataset paths
# absolute path for raw and processed folders
# dataset_type: "raw" or "processed"
def get_dataset_path(dataset_type: str) -> str:
    switch = {
        "raw": f"{DATASET_FOLDER_ABS_PATH}/raw",
        "processed": f"{DATASET_FOLDER_ABS_PATH}/processed"
    }
    return switch[dataset_type]

# Output paths
# absolute path for figures and reports folders
# output_type: "figures" or "reports"
def get_output_path(output_type: str) -> str:
    switch = {
        "figures": f"{OUTPUT_FOLDER_ABS_PATH}/figures",
        "reports": f"{OUTPUT_FOLDER_ABS_PATH}/reports"
    }
    return switch[output_type]
