
# Development

The best approach for doing development is to make changes within the repo and then build a new version of the GamutRF Docker container. To do that, run this docker command in the main directory of the repo:

```bash
docker build -t iqtlabs/gamutrf:latest .
```

Since it is tagged with **iqtlabs/gamutrf:latest** the docker-compose file (orchestrator.yml) will use the local image you just built instead of the one from docker hub. If you want to revert back to using the docker hub version simply run:

```bash
docker rmi iqtlabs/gamutrf:latest
```

This will of course generate lots of container images. To clean up old images, occasionally run:

```bash
docker system prune
```

### gr-iqtlabs Development

If you are looking to make changes to [gr-iqtlabs](https://github.com/IQTLabs/gr-iqtlabs) and then test them inside GamutRF, the following approach should work. The gr-iqtlabs libraries are pulled in and built in a base Dockerfile, so some changes are needed.

First, go and clone a local copy of the **gr-iqtlabs** repo. Make any changes you want. Because the **gr-iqtlabs** folder needs to be in the build context for the GamutRF base Dockerfile, you need to do some weird stuff. 

In the **gamutrf** repo open the `docker/Dockerfile.base` file. Line 31 `RUN git clone https://github.com/iqtlabs/gr-iqtlabs -b 1.0.76` can be commented out.

Line 32 `COPY --from=iqtlabs/gamutrf-vkfft:latest /root /root/gr-iqtlabs` needs to be updated to copy the files from your local copy of the **gr-iqtlabs** repo instead. The build will actually be done in the **gr-iqtlabs** folder, so make this change.
```docker
COPY . /root/gr-iqtlabs
```

Now make the following call while in the **gr-iqtlabs** repo folder, updated to reflect the correct path to the **Dockerfile.base** file:

```bash
docker build -f ../gamutRF/docker/Dockerfile.base -t iqtlabs/gamutrf-base:latest .
```

This will create a local version of the **gamutRF** base Dockerfile.

After this is done you will need to create a new version of the local gamutrf:latest image. Go back to the **gamutrf** repo directory and run the following command:

```bash
docker build -t iqtlabs/gamutrf:latest .
```

Startup the `docker compose up` command and your changes to **gr-iqtlabs** should be included.



## Dockerfile structure

- Dockerfile: builds the **gamutrf** container
- docker/: The Dockerfiles used to **build** the containers used in GamutRF
   - Dockerfile.base: creates the build containers that are used along with the base container.
   - Dockerfile.driver: build container for the different Soapy Drivers used to support the SDRs
   - Dockerfile.uhd-sr: build container for the UHD sample recorder



# Code Structure

- gamutrf Docker Container  (./Dockerfile)
  - gamurtrf/scan.py -> gamutrf-scan (/gamutrf/scan.py): parses all of the arguments
    - gamutrf/grscan.py:


# grscan

 - __init__()
   - find freq_range (end - start)
   - find fft_rate (samp_rate / nfft)
   - sets the *tune_step_fft*, which will default to *nfft* which is 2048
   - creates the source by calling **get_sources()** which is in **gamutrf/grsource.py**
     - sets up different SDR sources - Soapy, Ettus, file, etc
   - calls **get_fft_blocks()** 
      - **get_offload_fft_blocks()**
        - loads one of 3 FFT blocks: wavelearner for AIR-T, IQT Labs vkFFT block or GR FFT_VCC block
      - **get_pretune_block()**
        - if pretuning is used it will use the IQT Labs **retune_pre_fft()** block, otherwise it will use the GR stream_to_vector block
   - Adds **get_dc_blocks()** before the FFT block
     - if the *dc_block_len* variable is set, it will return the GR **dc_blocker_cc** block
   - Adds **get_db_blocks** after the FFT block
     - Adds a series of GR blocks to scale the output of the 
   - if **write_samples** variable is set, it will add the IQT Labs **write_freq_samples()**
   - Adds the IQT Labs **retune_fft()** block from the **gr-iqtlabs** repo to the end of the FFT set of blocks
   - Adds a ZeroMQ sink to the end of the FFT set of blocks
   - If the *inference_output_dir* variable is set it creates an IQT Labs **image_inference** block from the **gr-iqtlabs** repo
     - If the *mqtt_server* variable is defined it will add an **inference2mqtt** block from the **gamutrf/grinference2mqtt.py** file
   - if *pretune* variable is defined it will wire up the message ports
     - if *pretune* is enabled, the **retune_pre_fft** block will be connected to the source and the **retune_fft** block
     - if not, the **retune_fft** block is connected to the source
   - it then connects the blocks together using a mix of standard GR and a function which pulls from list of blocks **connect_blocks**

   The construction flow graph should be:

    - SOURCE **Tuneable File Source** or **Ettus** or **Soapy Source**
    - *DC Block*
    - FFT **wavelearner** or **VKFFT** or **FFT_VCC**
    - *retune_pre_fft()* pretuning
    - DB Block - FFT scaling
    - **retune_fft** IQT Labs gr-iqtlabs

   Attached to **retune_fft**

    - Inference Blocks
      - **image_inference**
      - *inference2mqtt*


    - Zero MQ Pub Sink
      - **zeromq.pub_sink()**

    - Writing the captured samples to file
      - *write_freq_samples()*


# retune_pre_fft

**part of gr-iqtlabs repo**

## Code Structure

- **general_work()**
  - **get_tags()** is in base_impl.cc, it goes through the received tags and pulls the tags that match along with the time and puts them in the array
  - if no tags were received, goto **process_items**
  - else pull out the first Freq/Time from the tags and then goto **process_items**
    - if you are not in *low_power_hold* or *stare_mode*
    - if *pending_retune_* then reduce it by 1, store some state stuff


- **process_items()**
  - loop through all of the samples, incrementing by the size of the FFT (*nfft_*)
    - if you are getting non-zero packets, and you have not hit the target number of packets, copy 1 NFFT worth of packets to output
    - call **need_retune_()** in **retuner_impl** to keep track of how many FFTs have been processed is greater than *tune_step_fft_*
      - if it is, signal that a retune is needed
    
# retune_fft

**part of gr-iqtlabs repo**

- **general_work()**

 - check if the output buffer is NOT empty, move what ever is there to a leftover buffer, then delete stuff 🤷
 - call **process_tags_**
   - get all the tags, pull out the ones we like by calling **get_tags()** from **base_impl.cc**
   - pull info from tags and propigate if opproriate
   - call **process_items_()**
     - since this is FFT values, collect the max, to find if the Ettus radio is in a low power mode after retuning. This lets you know if the retuning has completed and you do not have to wait a static amount of time ( this was previously done with the *skip_fft_count* )
     - loop through an copy over *nfft* worth of samples at a time
     - once we hit the target number of FFTs (the amount we should dwell at that tuning) tune to the next increment
   - then call **process_buckets_**
     - sees if there is a new freq, and sets up a new file
     - calls **write_bukets_** which writes all of the FFTs? 🤷 to a file


# image_inferece

**part of gr-iqtlabs repo**

## Code Structure

- **general_work()**
  - get all the tags, pull out the ones you want
  - propigate tags if you have them
  - if there are any JSON objects from processed inference, pop them from the queue and write them to *output*
  - call **process_items_()**
    - go through all of the items and generate rows of values
    - when you have the targetted number of rows call **create_image()**
      - if there are enough points above a certain level then add the image/meta to *inference_q_*

- **background_run_inference_**
  - This is runs in a thread that gets started up when the block is created
  - it is a simple loop that sleeps and calls **run_inference_()**
    - while the *inference_q_* is not empty go through this:
      - pop an item from the queue
      - build a JSON object with the metadata
      - do some light flips on the image to get it in the right orientation
      - send an HTTP request to Torch Serve
      - get the result, which should have the detections
      - call **parse_inference_()** to overlay the detections
      - call **write_image_** to create a file with the detections overlayed
        - 




# Notes
- wavelearner: this is the FFT library used for the AIR-T SDR
- low_power_hold_down:
This is the infamous ettus low power workaround. The ettus driver immediately acks and tags a retune command (well almost immediately)

However even after the tag it continues to emit samples that are for the old frequency. Then, the ettus enters a state where it outputs all zeros, then it starts emitting samples again at the new frequency

We send a retune and immediately enter holddown, discarding samples. We wait until we see all zeros. Then we resume sending samples again