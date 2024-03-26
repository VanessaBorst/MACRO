def calc_same_padding_for_stride_one(dilation, kernel_size):
    return (dilation * (kernel_size - 1) + 1) // 2

