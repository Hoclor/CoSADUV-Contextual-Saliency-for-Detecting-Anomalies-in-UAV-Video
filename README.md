# Contextual Saliency for Detecting Anomalies within Unmanned Aerial Vehicle (UAV) Video

Created and tested using Python 3.5.3, [PyTorch 0.4.0](https://pytorch.org/), and [OpenCV 3.4.3](https://opencv.org/)

Repository for my Master's Project on the topic of "Contextual Saliency for Detecting Anomalies within Unmanned Aerial Vehicle (UAV) Video". Paper available [here](https://github.com/Hoclor/CoSADUV-Contextual-Saliency-for-Detecting-Anomalies-in-UAV-Video/blob/master/Contextual_Saliency_for_Detecting_Anomalies_within_Unmanned_Aerial_Vehicle_(UAV)_Video.pdf).

Also code repository for our paper ["An Evaluation of Temporal and Non-Temporal Contextual Saliency Analysis for Generalized Wide-Area Search within Unmanned Aerial Vehicle Video"](https://github.com/Hoclor/CoSADUV-Contextual-Saliency-for-Detecting-Anomalies-in-UAV-Video/blob/master/Temporal_Contextual_Saliency_for_Wide_Area_Search_in_UAV_Video.pdf), under review for ICRA2020.

## Architectures

![Original DSCLRCN Architecture](https://github.com/Hoclor/CoSADUV-Contextual-Saliency-for-Detecting-Anomalies-in-UAV-Video/blob/master/images/CoSADUV.png "Our proposed CoSADUV architecture")

The DSCLRCN architecture ([original authors (and original image source)](https://github.com/nian-liu/DSCLRCN), [re-implementation in PyTorch used for this project](https://github.com/AAshqar/DSCLRCN-PyTorch)) was used as a baseline. Our proposed CoSADUV architecture is shown in the figure above, with changes from the DSCLRCN architecture shown with a grey background.

The architecture was modified by replacing the "Conv+Softmax" with a [Convolutional-LSTM](https://github.com/ndrplz/ConvLSTM_pytorch) layer with kernel size 3x3 and a Sigmoid activation function. Additionally, several loss functions other than NSSLoss were investigated (see our [paper](https://github.com/Hoclor/CoSADUV-Contextual-Saliency-for-Detecting-Anomalies-in-UAV-Video/blob/master/paper.pdf) for more information). Architectures with the convolutional LSTM (CoSADUV) and without it (CoSADUV_NoTemporal, using a normal conv layer instead) are available.

## Abstract

*"Unmanned Aerial Vehicles (UAV) can be used to great effect for the purposes of surveillance or search and rescue operations. UAV enable search and rescue teams to cover large areas more efficiently and in less time. However, using UAV for this purpose involves the creation of large amounts of data (typically video) which must be analysed before any potential findings can be uncovered and actions taken. This is a slow and expensive process which can result in significant delays to the response time after a target is seen by the UAV. To solve this problem we propose a deep model using a visual saliency approach to automatically analyse and detect anomalies in UAV video. Our Contextual Saliency for Anomaly Detection in UAV Video (CoSADUV) model is based on the state-of-the-art in visual saliency detection using deep convolutional neural networks and considers local and scene context, with novel additions in utilizing temporal information through a convolutional LSTM layer and modifications to the base model. Our model achieves promising results with the addition of the temporal implementation producing significantly improved results compared to the state-of-the-art in saliency detection. However, due to limitations in the dataset used the model fails to generalize well to other data, failing to beat the state-of-the-art in anomaly detection in UAV footage. The approach taken shows promise with the modifications made yielding significant performance improvements and is worthy of future investigation. The lack of a publicly available dataset for anomaly detection in UAV video poses a significant roadblock to any deep learning approach to this task, however despite this our paper shows that leveraging temporal information for this task, which the state-of-the-art does not currently do, can lead to improved performance."*

## Instructions to run the model
1. First follow all instructions in the READMEs inside model/ and model/Dataset/ to download and prepare the pretrained models used and to format the dataset correctly. (This may require the use of the scripts in data_preprocessing/)
2. Open and have a look through either `main_ncc.py` or `notebook_main.ipynb` (or both). The IPython notebook provides a more interactive interface, while `main_ncc.py` can be run with minimal input required.
3. Set the hyperparameters and settings near the top of the file. Most important is to set the correct dataset directory name (if different) and mean image filename.
4. Run through the file to train the model. If using `main_ncc.py`, the model will automatically be run through testing once training is completed.

## Example Results


[![Example](https://github.com/Hoclor/CoSADUV-Contextual-Saliency-for-Detecting-Anomalies-in-UAV-Video/blob/master/images/person7_thumbnail.PNG)](https://youtu.be/9qyMTolKbqc)

Click the video above to play it. The video shows the performane of the model on a sequence from the [UAV123](https://uav123.org/) dataset (This was also used to train the model). It presents four streams simultaneously:
- top-left is the input sequence
- top-right is the ground-truth data (as provided in the UAV123 dataset)
- bottom-left is the prediction of the best non-temporal model, and
- bottom-right is the prediction of the best temporal model.

## Reference

If you make use of this work in any way, please reference the following:

```
@MastersThesis{SimonGokstorp2019,
    author     =     {Simon Gokstorp},
    title     =     {{Contextual Saliency for Detecting Anomalies withing Unmanned Aerial Vehicle (UAV) Video}},
    school     =     {Durham University},
    address     =     {United Kingdom},
    year     =     {2019},
    }
```

If you have any questions/issues/ideas, feel free to open an issue here or contact me!

## Acknowledgements

This project was completed under the supervision of [Professor Toby Breckon](https://github.com/tobybreckon) (Durham University).
