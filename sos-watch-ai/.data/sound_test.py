import soundata

# Load a dataset
urbansound8k = soundata.initialize('urbansound8k')

# Download the dataset (only metadata if `partial_download=True`)
urbansound8k.download(partial_download=["index"])  # or ["all"] for full download

# Load a clip and view its metadata
clip = urbansound8k.clip_ids[0]
clip_data = urbansound8k.clip(clip)
print(clip_data)
