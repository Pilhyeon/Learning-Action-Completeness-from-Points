# Learning-Action-Completeness-from-Points
### Official Pytorch Implementation of '[Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization](https://arxiv.org/abs/2108.05029)' (ICCV 2021 Oral)

![architecture](https://user-images.githubusercontent.com/16102333/128889635-f07218d9-770a-4ece-a384-656e00b2794e.png)

> **Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization**<br>
> Pilhyeon Lee (Yonsei Univ.), Hyeran Byun (Yonsei Univ.)
>
> Paper: https://arxiv.org/abs/2108.05029
>
> **Abstract:** *We tackle the problem of localizing temporal intervals of actions with only a single frame label for each action instance for training. Owing to label sparsity, existing work fails to learn action completeness, resulting in fragmentary action predictions. In this paper, we propose a novel framework, where dense pseudo-labels are generated to provide completeness guidance for the model. Concretely, we first select pseudo background points to supplement point-level action labels. Then, by taking the points as seeds, we search for the optimal sequence that is likely to contain complete action instances while agreeing with the seeds. To learn completeness from the obtained sequence, we introduce two novel losses that contrast action instances with background ones in terms of action score and feature similarity, respectively. Experimental results demonstrate that our completeness guidance indeed helps the model to locate complete action instances, leading to large performance gains especially under high IoU thresholds. Moreover, we demonstrate the superiority of our method over existing state-of-the-art methods on four benchmarks: THUMOS'14, GTEA, BEOID, and ActivityNet.
Notably, our method even performs comparably to recent fully-supervised methods, at the 6 times cheaper annotation cost.*


## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.6
* Tensorflow 1.15 (for Tensorboard)
* CUDA 10.2

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We excluded three test videos (270, 1292, 1496) as previous work did.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used. You can find them [here](https://drive.google.com/file/d/1NqaDRo782bGZKo662I0rI_cvpDT67VQU/view?usp=sharing).
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── point_gaussian
           └── point_labels.csv
       └── features
           ├── train
               ├── rgb
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
               └── flow
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
           └── test
               ├── rgb
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
               └── flow
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
~~~~

## Usage

### Running
You can easily train and evaluate the model by running the script below.

If you want to try other training options, please refer to `options.py`.

~~~~
$ bash run.sh
~~~~

### Evaulation
The pre-trained model can be found [here](https://drive.google.com/file/d/1xH-dH2UaR7hoYK6MRsc_29mwU0zJCDzU/view?usp=sharing).
You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We note that this repo was built upon our previous models.
* Background Suppression Network for Weakly-supervised Temporal Action Localization (AAAI 2020) [[paper](https://arxiv.org/abs/1911.09963)] [[code](https://github.com/Pilhyeon/BaSNet-pytorch)]
* Weakly-supervised Temporal Action Localization by Uncertainty Modeling (AAAI 2021) [[paper](https://arxiv.org/abs/2006.07006)] [[code](https://github.com/Pilhyeon/WTAL-Uncertainty-Modeling)]

We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [SF-Net](https://github.com/Flowerfan/SF-Net)
* [ActivityNet](https://github.com/activitynet/ActivityNet)

In addition, we referenced a part of code in the following repo for the greedy algorithm implementation.

* [NeuralNetwork-Viterbi](https://github.com/alexanderrichard/NeuralNetwork-Viterbi)

## Citation
If you find this code useful, please cite our paper.

~~~~
@inproceedings{lee2021completeness,
  title={Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization},
  author={Pilhyeon Lee and Hyeran Byun},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2021},
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Pilhyeon Lee (lph1114@yonsei.ac.kr).
