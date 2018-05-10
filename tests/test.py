import os
import numpy as np

import cluda

def test_add():
    n = 1000
    float_type = np.float32

    a = np.random.rand(n)
    b = np.random.rand(1)[0]

    gpu_results = cluda.empty_gpu(n, float_type)
    gpu_in = cluda.to_gpu(a, float_type)

    block_size = 128
    n_blocks = int(np.ceil(n / block_size))

    here_dir = os.path.dirname(os.path.realpath(__file__))
    module = cluda.load_gpu(
        'add.cu',
        tmpl_args = dict(
            float_type = cluda.np_to_c_type(float_type),
            b = b,
            block_size = block_size
        ),
        tmpl_dir = here_dir
    )
    module.add(
        gpu_results, gpu_in, np.int32(n),
        grid = (n_blocks, 1, 1), block = (block_size, 1, 1)
    )
    out = gpu_results.get()

    np.testing.assert_almost_equal(out, a + b)
