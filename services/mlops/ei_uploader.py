import edgeimpulse as ei
import os

# get and set edge impulse api key from .env
EI_API_KEY = os.getenv("EI_API_KEY")
ei.API_KEY = EI_API_KEY

# get project id from .env
EI_PROJECT_ID = os.getenv("EI_PROJECT_ID")


# possibly fetch the manifest (list) the files already uploaded to edge impulse

# fetch the processed manifest

# upload processed files not yet uploaded to edge impulse
# eg:
# audio_file_path = "path/to/your/audio_sample.wav"
# label = "your_audio_label"
# category = "training" # or "testing" or "validation"

# response = ei.experimental.data.upload_sample(
#     filename=audio_file_path,
#     category=category,
#     label=label
# )
# print(f"Uploaded sample with ID: {response['id']}")

# double check the final EI manifest matches the processed manifest