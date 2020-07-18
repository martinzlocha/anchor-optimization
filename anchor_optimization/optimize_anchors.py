
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import csv
import os
import sys
import numpy as np
import scipy.optimize
from PIL import Image
from .utils.compute_overlap import compute_overlap
from keras_retinanet.preprocessing.csv_generator import _open_for_csv
from keras_retinanet.utils.anchors import generate_anchors, AnchorParameters, anchors_for_shape
from keras_retinanet.utils.image import compute_resize_scale


# global variable
global state
state = {'best_result': sys.maxsize}


def calculate_config(values,
                     ratio_count,
                     SIZES=[32, 64, 128, 256, 512],
                     STRIDES=[8, 16, 32, 64, 128]):

    split_point = int((ratio_count - 1) / 2)

    ratios = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1 / values[i])

    scales = values[split_point:]

    return AnchorParameters(SIZES, STRIDES, ratios, scales)


def base_anchors_for_shape(pyramid_levels=None,
                           anchor_params=None):

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        all_anchors = np.append(all_anchors, anchors, axis=0)

    return all_anchors


def average_overlap(values,
                    entries,
                    image_shape,
                    mode='focal',
                    ratio_count=3,
                    include_stride=False,
                    SIZES=[32, 64, 128, 256, 512],
                    STRIDES=[8, 16, 32, 64, 128],
                    verbose=False,
                    set_state=None,
                    to_tuple=False,
                    threads=1):

    anchor_params = calculate_config(values,
                                     ratio_count,
                                     SIZES,
                                     STRIDES)
    if include_stride:
        anchors = anchors_for_shape(image_shape, anchor_params=anchor_params)
    else:
        anchors = base_anchors_for_shape(anchor_params=anchor_params)

    overlap = compute_overlap(entries, anchors)
    max_overlap = np.amax(overlap, axis=1)
    not_matched = len(np.where(max_overlap < 0.5)[0])

    if mode == 'avg':
        result = 1 - np.average(max_overlap)
    elif mode == 'ce':
        result = np.average(-np.log(max_overlap))
    elif mode == 'focal':
        result = np.average(-(1 - max_overlap) ** 2 * np.log(max_overlap))
    else:
        raise Exception('Invalid mode.')

    if set_state is not None:
        state = set_state

    # --------------------------------------------------------------------------------------------------------------------------------
    # "scipy.optimize.differential_evolution" utilizes multiprocessing but internally uses "multiprocessing.Pool" and not
    # "multiprocessing.Process" which is required for sharing state between processes
    # (see: https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes)
    #
    # the "state" variable does not affect directly the "scipy.optimize.differential_evolution" process, therefore updates will be
    # printed out in case of improvement only if a single thread is used
    # --------------------------------------------------------------------------------------------------------------------------------

    if threads == 1:

        if result < state['best_result']:
            state['best_result'] = result

            if verbose:
                print('Current best anchor configuration')
                print('State: {}'.format(np.round(state['best_result'], 5)))
                print(
                    'Ratios: {}'.format(
                        sorted(
                            np.round(
                                anchor_params.ratios,
                                3))))
                print(
                    'Scales: {}'.format(
                        sorted(
                            np.round(
                                anchor_params.scales,
                                3))))

            if include_stride:
                if verbose:
                    print(
                        'Average overlap: {}'.format(
                            np.round(
                                np.average(max_overlap),
                                3)))

            if verbose:
                print(
                    "Number of labels that don't have any matching anchor: {}".format(not_matched))
                print()

    if to_tuple:
        # return a tuple, which happens in the last call to the 'average_overlap' function
        return result, not_matched
    else:
        return result



def anchors_optimize(annotations,
                     ratios=3,
                     scales=3,
                     objective='focal',
                     popsize=15,
                     mutation=0.5,
                     image_min_side=800,
                     image_max_side=1333,
                     # default SIZES values
                     SIZES=[32, 64, 128, 256, 512],
                     # default STRIDES values
                     STRIDES=[8, 16, 32, 64, 128],
                     include_stride=False,
                     resize=False,
                     threads=1,
                     verbose=False,
                     seed=None):
    
    """
    Important Note: The python "anchors_optimize" function is meant to be used from the command line (from within a Python console it gives incorrect results)    
    """

    if ratios % 2 != 1:
        raise Exception('The number of ratios has to be odd.')

    entries = np.zeros((0, 4))
    max_x = 0
    max_y = 0

    updating = 'immediate'
    if threads > 1:
        # when the number of threads is > 1 then 'updating' is set to 'deferred' by default (see the documentation of "scipy.optimize.differential_evolution())
        updating = 'deferred'

    if seed is None:
        seed = np.random.RandomState()
    else:
        seed = np.random.RandomState(seed)

    if verbose:
        print('Loading object dimensions.')

    with _open_for_csv(annotations) as file:
        for line, row in enumerate(csv.reader(file, delimiter=',')):
            x1, y1, x2, y2 = list(map(lambda x: int(x), row[1:5]))

            if not x1 or not y1 or not x2 or not y2:
                continue

            if resize:
                # Concat base path from annotations file follow retinanet
                base_dir = os.path.split(annotations)[0]
                relative_path = row[0]
                image_path = os.path.join(base_dir, relative_path)
                img = Image.open(image_path)

                if hasattr(img, "shape"):
                    image_shape = img.shape
                else:
                    image_shape = (img.size[0], img.size[1], 3)

                scale = compute_resize_scale(
                    image_shape, min_side=image_min_side, max_side=image_max_side)
                x1, y1, x2, y2 = list(map(lambda x: int(x) * scale, row[1:5]))

            max_x = max(x2, max_x)
            max_y = max(y2, max_y)

            if include_stride:
                entry = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
                entries = np.append(entries, entry, axis=0)
            else:
                width = x2 - x1
                height = y2 - y1
                entry = np.expand_dims(
                    np.array([-width / 2, -height / 2, width / 2, height / 2]), axis=0)
                entries = np.append(entries, entry, axis=0)

    image_shape = [max_y, max_x]

    if verbose:
        print('Optimising anchors.')

    bounds = []

    for i in range(int((ratios - 1) / 2)):
        bounds.append((1, 4))

    for i in range(scales):
        bounds.append((0.4, 2))

    update_state = None
    if threads == 1:
        update_state = state

    ARGS = (entries,
            image_shape,
            objective,
            ratios,
            include_stride,
            SIZES,
            STRIDES,
            verbose,
            update_state,
            # return a single value ('to_tuple' parameter is set to False)
            False,
            threads)

    result = scipy.optimize.differential_evolution(func=average_overlap,
                                                   # pass the '*args' as a tuple (see: https://stackoverflow.com/q/32302654)
                                                   args=ARGS,
                                                   mutation=mutation,
                                                   updating=updating,
                                                   workers=threads,
                                                   bounds=bounds,
                                                   popsize=popsize,
                                                   seed=seed)

    if hasattr(result, 'success') and result.success:
        print('Optimization ended successfully!')
    elif not hasattr(result, 'success'):
        print('Optimization ended!')
    else:
        print('Optimization ended unsuccessfully!')
        print('Reason: {}'.format(result.message))

    values = result.x
    anchor_params = calculate_config(values,
                                     ratios,
                                     SIZES,
                                     STRIDES)

    (avg, not_matched) = average_overlap(values,
                                         entries,
                                         image_shape,
                                         'avg',
                                         ratios,
                                         include_stride,
                                         SIZES,
                                         STRIDES,
                                         verbose,
                                         # pass a specific value to the 'set_state' parameter
                                         {'best_result': 0},
                                         # return a 'tuple'  ('to_tuple' parameter is set to True)
                                         True,
                                         # set the 'threads' parameter to 1
                                         1)

    # as 'end_state' set the 'avg' value
    end_state = np.round(avg, 5)
    RATIOS_result = sorted(np.round(anchor_params.ratios, 3))
    SCALES_result = sorted(np.round(anchor_params.scales, 3))

    print()
    print('Final best anchor configuration')
    print('State: {}'.format(end_state))
    print('Ratios: {}'.format(RATIOS_result))
    print('Scales: {}'.format(SCALES_result))

    dict_out = {
        'ratios': RATIOS_result,
        'scales': SCALES_result,
        'not_matched': not_matched,
        'end_state': end_state}

    if include_stride:
        STRIDE = np.round(1 - avg, 3)
        print('Average overlap: {}'.format(STRIDE))
        dict_out['stride'] = STRIDE

    print("Number of labels that don't have any matching anchor: {}".format(not_matched))

    return dict_out
