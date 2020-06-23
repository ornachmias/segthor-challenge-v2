import numpy as np

def axis_based_denoise_method(model_results, categories):
    slices, x_length, y_length = model_results.shape
    block_in = {category: [] for category in categories}
    block_out = {category: [] for category in categories}
    for idx in range(slices):
        if idx == 0:
            previous_slice = np.array([0] * x_length * y_length)
        else:
            previous_slice = model_results[idx - 1, :, :]

        current_slice = model_results[idx, :, :]
        if idx + 1 == slices:
            next_slice = np.array([0] * x_length * y_length)
        else:
            next_slice = model_results[idx + 1, :, :]

        for category in categories:
            if category not in previous_slice and category in current_slice:
                block_in[category].append(idx)
            if category in current_slice and category not in next_slice:
                block_out[category].append(idx)

    model_results = model_results.copy()
    for category in categories:
        length = [_out - _in for _out, _in in zip(block_out[category], block_in[category])]
        if not length:
            continue
        idx = length.index(max(length))
        start = block_in[category][idx]
        end = block_out[category][idx]
        view = model_results.reshape(slices, -1)
        for sl in range(slices):
            if start <= sl <= end:
                continue
            view[sl][view[sl] == category] = 0
    return model_results


# #
# array = \
#     [
#         [
#             [1, 3, 1, 2],
#             [1, 1, 1, 1],
#         ],
#         [
#             [1, 1, 1, 2],
#             [1, 1, 1, 1],
#         ],
#         [
#             [3, 3, 3, 3],
#             [2, 3, 3, 3],
#         ],
#         [
#             [3, 1, 3, 2],
#             [2, 3, 3, 3],
#         ]
#
#     ]
# categories = [1, 2, 3, 4]
# print(axis_based_denoise_method(np.array(array), categories))
