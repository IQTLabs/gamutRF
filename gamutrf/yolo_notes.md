
# Train 
Define training parameters:
- /home/ltindall/RFClassification/params.yaml  
```
model_type: yolov8n.pt
pretrained: True
seed: 0
imgsz: 640
batch: 4
epochs: 100
optimizer: Adam # other choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
lr0: 0.01  # learning rate
name: 'yolov8s_exp_v0' # experiment name
```

- /home/ltindall/RFClassification/data/roboflow/data.yaml    
## Run
`(airstack-py39) ltindall@icebreaker:~/RFClassification$ python yolov8_train.py`

<br>

# Yolov8 training output
## Creates output in 
`/home/ltindall/ultralytics/runs/`

<br>

# Convert .pt to .onnx
## Run
`(airstack-py36) ltindall@icebreaker:~$ yolo export model=/home/ltindall/yolov8s_weights_6_7.pt format=onnx device=0`   
## Creates
`/home/ltindall/yolov8s_weights_6_7.onnx`

<br>

# Convert .onnx to .engine/.plan
## Run
`(airt-py39) ltindall@icebreaker:~/gamutrf$ /usr/src/tensorrt/bin/trtexec --onnx=/home/ltindall/yolov8s_weights_6_7.onnx --saveEngine=/home/ltindall/yolov8s_weights_6_7.plan`   
## Creates
`/home/ltindall/yolov8s_weights_6_7.plan`

<br>

# BROKEN: convert .pt to .engine/.plan
> Note: Converting straight from PyTorch .pt to .engine/.plan has not been working. Reasons unknown. Instead convert .pt->.onnx->.plan 

`(airstack-py36) ltindall@icebreaker:~$ yolo export model=/home/ltindall/yolov8s_weights_6_7.pt format=engine device=0`   
--> /home/ltindall/yolov8s_weights_6_7.engine

<br>

# Run scanner with Yolov8
## Run
`(airt-py39) ltindall@icebreaker:~/gamutrf$ gamutrf-scan --sdr=SoapyAIRT --freq-start=5.6e9 --freq-end=5.8e9 --tune-step-fft 2048 --samp-rate=100e6 --nfft 256 --tuneoverlap=1 --inference_plan_file=/home/ltindall/yolov8s_weights_6_7.plan --inference_output_dir=inference_output`