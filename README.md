# cluda
CLUDA -- Write once, run anywhere with both CUDA and OpenCL.

Here's a full example of usage with comments to guide you along! The example runs some "CLUDA" code that adds a prespecified number to every element in an array. Check the `tests` folder for the full example.

First, the "CLUDA". This is passed through the Mako templating engine along the way to being compiled, so you're actually able to have a few parameter in there ("float_type"). In addition the `cluda_preamble` declares a whole bunch of C macros to be the necessary value for either CUDA or OpenCL. These are then used later: `KERNEL`, `GLOBAL_MEM`. See the definition of `cluda_preamble` in `cluda/cuda.py` and `cluda/opencl.py` for the full list.
```
${cluda_preamble} 

#define Real ${float_type}

KERNEL
void add(GLOBAL_MEM Real* results, GLOBAL_MEM Real* a, int n) {
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    results[i] = a[i] + ${b};
}
```

```python

import os
import numpy as np

import cluda

def test_add():
    n = 1000
    float_type = np.float32

    a = np.random.rand(n)
    b = np.random.rand(1)[0]

    # Let's create a space for our kernel output!
    gpu_results = cluda.empty_gpu(n, float_type)
    
    # And a space for our kernel input!
    gpu_in = cluda.to_gpu(a, float_type)

    block_size = 128
    n_blocks = int(np.ceil(n / block_size))

    here_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Okay, this is the meat of it, let's load the 'add.cu' file and compile it.
    # The file itself is allowed to have Mako templates and any template arguments
    # are passed through the "tmpl_args" dictionary.
    module = cluda.load_gpu(
        'add.cu',
        tmpl_args = dict(
            float_type = cluda.np_to_c_type(float_type),
            b = b,
            block_size = block_size
        ),
        tmpl_dir = here_dir
    )
    
    # Now let's actually call our "CLUDA" add function.
    module.add(
        gpu_results, gpu_in, np.int32(n),
        grid = (n_blocks, 1, 1), block = (block_size, 1, 1)
    )
    
    # And copy the result from GPU memory back to the CPU.
    out = gpu_results.get()

    # Check that it's correct.
    np.testing.assert_almost_equal(out, a + b)
```

The interface is pretty minimal, but it's been enough for everything I've used it for so far and the fundamental idea has been incredibly helpful.
