Light Weight Face Liveness Neural Network for Face Spoofing Detection

Overview

This repository contains a lightweight dual-stream CNN architecture designed for efficient image classification: face spoofing or face presentation attack detection. 
The model balances accuracy and computational efficiency, making it suitable for deployment on resource-constrained devices.

Architecture

The architecture consists of two parallel streams:

- Stream 1: A shallow CNN with two convolution layers followed by a Dropout layer.
- Stream 2: A deeper CNN with three convolution layers and a combination of Dense and dropout layers.

The outputs from both streams are fused using a concatenation layer, followed by a final classification layer.

Features

- Lightweight: Model size reduced by 30% compared to similar architectures.
- Dual-stream: Improved accuracy and robustness through multi-scale feature fusion.
- Efficient: Optimized for detection of both 2D and 3D attacks

Usage

1. 3D MAD Dataset - https://www.idiap.ch/en/scientific-research/data/3dmad. publicly available after signing the EULA Agreement
2. NUAA Dataset - https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html. available for use freely used the third format available on link for the proposed work.
3. Video Replay Attack Dataset - https://www.idiap.ch/en/scientific-research/data/replayattack.  publicly available after signing the EULA Agreement

Training

1. Prepare dataset: for 3D MAD dataset use every 10th position image from the real and fake images folder , Video Replay attack dataset use the train set and convert the videos to images, NUAA dataset rename the images and use all the images. 
2. Train model: Use the Architecture and train the model with above datasets with Train and val set and test it with the available test sets.



