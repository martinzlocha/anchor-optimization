## Introduction

Anchor optimization for RetinaNet based on [Improving RetinaNet for CT Lesion Detection with Dense Masks from Weak RECIST Labels](https://arxiv.org/abs/1906.02283) by Martin Zlocha, Qi Dou and Ben Glocker.
Implementation makes heavy use of [keras-retinanet](https://github.com/fizyr/keras-retinanet).

For questions and discussion join the [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) and either message me directly (username: martinzlocha) or join the `#keras-retinanet` channel.

## Setup

1. Clone this repository
1. `pip install .`
1. `python setup.py build_ext --inplace`

## Usage

Basic usage:

1. Define your own dataset in a csv format, for more information follow the guide in [keras-retinanet](https://github.com/fizyr/keras-retinanet#csv-datasets).
1. Run `anchor-optimization PATH_TO_CSV` 

Additional options:

- `scales` and `ratios` parameters allows you to define the number scales and ratios in your anchor configuration.
- `objective` parameter allows you to specify how the objective function is calculated. The `avg` parameter optimizes the average overlap between the anchors and the ground truth. We suggest that you use the `focal` parameter (default) instead because it ensures that there are less ground truth lesions with lower than 0.5 IoU with an anchor.
- `popsize` parameter allows you to specify the population multiplier used by differential evolution. Higher values result in better results however the computation will also be slower.
- By default the images and the detected objects are resized using the same method as in keras-retinanet implementation. If you do not wish to resize the image use the `no-resize` flag or specify different `image-min-side` and `image-max-side`.
- By default we ignore strides when optimizing the anchor configuration, this makes it feasible to optimize the anchors for a large number of objects in a short time. We have recently added an `include-stride` flag. This makes the computation much slower however it is more accurate. We suggest you only use it if your dataset is small.
- `threads` to allow parallelization.

To reproduce our results:

`anchor-optimization PATH_TO_CSV --ratios=5 --no-resize`

### Notes 

- This repository has been tested on python 3.7.
- This repository relies on keras-retinanet 0.5.1.

Contributions to this repository are welcome.

## Citation

If you find the code or the optimized anchors useful for your research, please consider citing our paper.

```
@article{zlocha2019improving,
  title={Improving RetinaNet for CT Lesion Detection with Dense Masks from Weak RECIST Labels},
  author={Zlocha, Martin and Dou, Qi and Glocker, Ben},
  journal={arXiv preprint arXiv:1906.02283},
  year={2019}
}
```
