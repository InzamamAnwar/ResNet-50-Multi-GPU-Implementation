# ResNet-50 Multi-GPU Implementation on TensorFlow

This repository presents a simple to understand and easy to follow guide of implementing ResNet-50 in TensorFlow. It is recommended to first study ResNet paper avialable [here.](https://arxiv.org/abs/1512.03385)


Input to the network is fed through [TFRecord](https://www.tensorflow.org/guide/datasets) file, made separately for training and testing.



All the parameters are placed in ``` param.py```

## Getting Started
```
    1. Clone or donwload the repository on local machine
    2. Copy train.tfrecords and test.tfrecords file in the directory
    3. Modify model/training parameters in param.py file
    4. Run multi_gpu_train.py file to start training
    5. After training, run ResNet_eval.py for test/eval
