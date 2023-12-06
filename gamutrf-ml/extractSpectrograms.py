from base64 import standard_b64decode
import zstandard
import bz2
import gzip
import os
import subprocess
import boto3
from pathlib import Path
import sigmf
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_reader(filename):
    # nosemgrep:github.workflows.config.useless-inner-function
    def bz2_reader(x):
        return bz2.open(x, "rb")

    # nosemgrep:github.workflows.config.useless-inner-function
    def gzip_reader(x):
        return gzip.open(x, "rb")

    # nosemgrep:github.workflows.config.useless-inner-function
    def zst_reader(x):
        return zstandard.ZstdDecompressor().stream_reader(
            open(x, "rb"), read_across_frames=True
        )

    def default_reader(x):
        return open(x, "rb")

    if filename.endswith(".bz2"): 
        return bz2_reader
    if filename.endswith(".gz"):
        return gzip_reader
    if filename.endswith(".zst"):
        return zst_reader

    return default_reader


#def read_samples(filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=None, fft_count=None):
def read_samples(filename, sample_dtype, seek_bytes=0, nfft=None, fft_count=None):
    print("Reading "+filename)
    reader = get_reader(filename)

    with reader(filename) as infile:
        infile.seek(int(seek_bytes))

        # Infer sample length from sigmf datatype string
        if sample_dtype == "cf32_le" or sample_dtype == "cf32" or sample_dtype == "cf32_be":
            sample_bytes = 8
        elif sample_dtype == "ci16_le" or sample_dtype == "ci16" or sample_dtype == "ci16_be" or sample_dtype == 'ri16_le':
            sample_bytes =  4
        elif sample_dtype == "ci8" or sample_dtype == "i8":
            sample_bytes = 2
        else:
            raise ValueError("Datatype " + sample_dtype + " not implemented")

        if fft_count is not None: 
            sample_buffer = infile.read(fft_count * nfft * sample_bytes)
        else: 
            sample_buffer = infile.read()

        buffered_samples = int(len(sample_buffer) / sample_bytes)

        if buffered_samples == 0:
            print("Error! No samples read from "+filename)
            return None
        if fft_count is not None and buffered_samples / nfft != fft_count:
            print("Incomplete sample file. Could not load the expected number of samples.")

        # Infer sample datatype from sigmf datatype string
        if sample_dtype == "ci16_le" or sample_dtype == "ci16" or sample_dtype == "ci16_be":
            samples = np.frombuffer(sample_buffer, dtype=np.int16)
            samples = samples[::2] + 1j * samples[1::2]
        elif sample_dtype == "cf32_le" or sample_dtype == "cf32" or sample_dtype == "cf32_be":
            samples = np.frombuffer(sample_buffer, dtype=np.complex64)
        elif sample_dtype == "ci8" or sample_dtype == "i8":
            samples = np.frombuffer(sample_buffer, dtype=np.int8)
            samples = samples[::2] + 1j * samples[1::2]
        elif sample_dtype == 'ri16_le':
            samples = np.frombuffer(sample_buffer, dtype=np.int16)
            
        else:
            raise ("Datatype " + sample_dtype + " not implemented")

        
        return samples

def prepare_custom_spectrogram(min_freq, max_freq, sample_rate, nfft, fft_count, noverlap):  
    freq_resolution = sample_rate / nfft
    max_idx = round((max_freq - min_freq) / freq_resolution)
    total_time = (nfft * fft_count) / sample_rate
    expected_time_bins = int((nfft * fft_count) / (nfft - noverlap))
    X, Y = np.meshgrid(
        np.linspace(
            min_freq,
            max_freq,
            int((max_freq - min_freq) / freq_resolution + 1),
        ),
        np.linspace(0, total_time, expected_time_bins),
    )
    spectrogram_array = np.empty(X.shape)
    spectrogram_array.fill(np.nan)
    
    return spectrogram_array, max_idx, freq_resolution


# Define the S3 bucket and file paths
s3_bucket = "gamutrf-anom-wifi"
s3_folder_paths = ["normal-wifi/cf32"]


# Create an S3 client
s3 = boto3.client('s3')

# Loop through the file paths
for s3_folder_path in s3_folder_paths:

    # Define the local directory to store the downloaded files
    local_directory = "extract-" + s3_bucket + "-" + s3_folder_path
    if not os.path.exists(local_directory):
       os.makedirs(local_directory)

    # List the objects in the folder
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder_path)
    
    # Extract the file names
    file_names = []
    for obj in response['Contents']:
        file_names.append(obj['Key'].split('/')[-1])  # Get the file name from the object key
    
    # Download the files in the folders
    for file_name in file_names:
        if Path(file_name).suffix == '.sigmf-data' or Path(file_name).suffix == '.sigmf-meta' or Path(file_name).suffix == '.zst':
            if not os.path.exists(local_directory + "/" + file_name):
                s3_download_command = f"aws s3 cp s3://{s3_bucket}/{s3_folder_path}/{file_name} {local_directory}"
                subprocess.run(s3_download_command, shell=True)

# Get values from sigmf meta file
sigmf_meta_files = [ fi for fi in file_names if fi.endswith(".sigmf-meta") ]
data_files = [ fi for fi in file_names if not fi.endswith(".sigmf-meta") ]
for sigmf_meta_file in sigmf_meta_files:
    handle = sigmf.sigmffile.fromfile(local_directory + "/" + sigmf_meta_file)
    data_file = list(filter(lambda x: x.startswith(Path(sigmf_meta_file).stem), data_files)) 
    print(data_file)
    
    globals = handle.get_global_info() # returns 'global' dictionary
    captures = handle.get_captures() # returns list of 'captures' dictionaries
    annotations = handle.get_annotations() # returns list of all annotation    

    #sample_dir = args.sample_dir
    nfft = 1024
    fft_count = 256
    save_data = True
    skip_inference = True
    center_frequency = captures[0]["core:frequency"]
    sample_rate = globals["core:sample_rate"]
    min_freq = center_frequency - ( sample_rate / 2)
    max_freq = center_frequency + ( sample_rate / 2)
    noverlap = 0
    cmap = plt.get_cmap("turbo")
    data_type = globals["core:datatype"]
    dtype = None
    length = None

    
    
    if (min_freq is None and max_freq is not None) or (min_freq is not None and max_freq is None): 
        print("Error! If min_freq or max_freq is defined then both must be defined. Exiting.")
        exit()
    if min_freq is not None and max_freq is not None: 
        custom_spectrogram = True
    else: 
        custom_spectrogram = False

    seek = 0
    sample_count = 0
    while sample_count < 20:
        #seek = seek + nfft*fft_count
        #samples = handle.read_samples(start_index=seek, count=nfft*fft_count) # returns all timeseries data
        
        samples = read_samples(
                        local_directory + "/" + data_file[0], 
                        data_type, 
                        #length, 
                        seek_bytes=seek, 
                        nfft=nfft, 
                        fft_count=fft_count,
                    )
        if samples is None:
                    print("Continuing...")
                    break
        
        # Convert samples into spectrogram
        freq_bins, t_bins, spectrogram = signal.spectrogram(
                        samples,
                        sample_rate,
                        window=signal.hann(int(nfft), sym=True),
                        nperseg=nfft,
                        noverlap=noverlap,
                        detrend='constant',
                        return_onesided=False,
                    )
        # FFT shift 
        freq_bins = np.fft.fftshift(freq_bins)
        spectrogram = np.fft.fftshift(spectrogram, axes=0)
    
        # Transpose spectrogram
        spectrogram = spectrogram.T
    
        # dB scale spectrogram
        spectrogram = 10 * np.log10(spectrogram)
    
        # Normalize spectrogram
        spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram)) #(spectrogram - db_min) / (db_max - db_min)
        spectrogram_data = spectrogram_normalized

        if custom_spectrogram: 
            if fft_count is None: 
                fft_count = len(t_bins)
                spectrogram_data, max_idx, freq_resolution = prepare_custom_spectrogram(
                    min_freq, 
                    max_freq, 
                    sample_rate, 
                    nfft, 
                    fft_count, 
                    noverlap
                )
                idx = np.array(
                            [
                                round((item - min_freq) / freq_resolution)
                                for item in freq_bins + center_frequency
                            ]
                        ).astype(int)
                spectrogram_data[
                            : spectrogram_normalized.shape[0],
                            idx[np.flatnonzero((idx >= 0) & (idx <= max_idx))],
                ] = spectrogram_normalized[:, np.flatnonzero((idx >= 0) & (idx <= max_idx))]

        # Spectrogram color transforms 
    
        #spectrogram_color = cv2.resize(cmap(spectrogram_data)[:,:,:3], dsize=(1640, 640), interpolation=cv2.INTER_CUBIC)[:,:,::-1]
        spectrogram_color = cmap(spectrogram_data)[:,:,:3] # remove alpha dimension
        spectrogram_color = spectrogram_color[::-1,:,:] # flip vertically
        spectrogram_color *= 255
        spectrogram_color = spectrogram_color.astype(int)
        spectrogram_color = np.ascontiguousarray(spectrogram_color, dtype=np.uint8)

        # Save spectrogram as .png
        if save_data: 
            spectrogram_img = Image.fromarray(spectrogram_color)
            image_dir = Path(f"{local_directory}/png-256/")
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{Path(sigmf_meta_file).stem}{sample_count}.png"
            spectrogram_img.save(image_path)
            #["img_file"] = str(image_path)
            print("Saved image to "+str(image_path))

        sample_count = sample_count + 1
        seek = seek + nfft*fft_count

    # Save metadata as .json 
    # meta_data["id"] = spectrogram_id
    # file_info["nfft"] = nfft
    # meta_data["sample_file"] = file_info
    #     meta_dir = Path(f"{sample_dir}/metadata/")
    #     meta_dir.mkdir(parents=True, exist_ok=True)
    #     json_object = json.dumps(meta_data, indent=4, cls=DtypeEncoder)
    #     meta_data_path = meta_dir / f"{basefilename}{sample_count}.json"
    #     with open(meta_data_path, "w") as outfile:
    #                     outfile.write(json_object)
    #     print("Saved metadata to "+str(meta_data_path))

    #                 # Run inference model
    #                 if not skip_inference:
    #                     if spectrogram_id > 0: # bug in yolov8, name parameters is broken in predict()
    #                         model.predictor.save_dir = Path(f"{sample_dir}/predictions/{basefilename}")
    #                     results = model.predict(source=spectrogram_color[:,:,::-1], conf=0.05, save=True, save_txt=True, save_conf=True, project=f"{sample_dir}/predictions/", name=f"{basefilename}{sample_count}", exist_ok=True)

    #                 spectrogram_id += 1    
    #                 seek = seek + nfft*fft_count
    #                 sample_count = sample_count + 1

    #exit()


# noverlap = 0 #nfft // 8
# model = YOLO("/home/ubuntu/gamutrf-ml/best.pt")

# spectrogram_id = 0 
# processed_files = []

# wait_count = 0 
# wait_time = 1
# wait_count_limit = 5
    
