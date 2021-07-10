def conv_output_dimension(input_size, kernel_size, stride, padding, dilation):
    # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    output_size = ((input_size + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1)
    assert int(((input_size + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1) == output_size
    return output_size


def calc_padding(desired_output_size, input_size, stride, dilation, kernel_size):
    # (window_size - (input_size - 1) * stride - dilation * (kernel_size - 1) - 1) / (-2)
    return int(((desired_output_size - 1) * stride - input_size + dilation * (kernel_size - 1) + 1) / 2)


def calc_same_padding_for_stride_one(dilation, kernel_size):
    return (dilation * (kernel_size - 1) + 1) // 2

