import os
import pyopencl
import pyopencl.array
import warnings
import numpy as np

import logging
logger = logging.getLogger(__name__)

gpu_initialized = False
gpu_ctx = None
gpu_queue = None

def report_devices(ctx):
    device_names = [d.name for d in ctx.devices]
    logger.info('initializing opencl context with devices = ' + str(device_names))

def initialize_with_ctx(ctx):
    global gpu_initialized, gpu_ctx, gpu_queue
    gpu_ctx = ctx
    gpu_queue = pyopencl.CommandQueue(
        gpu_ctx,
        properties=pyopencl.command_queue_properties.PROFILING_ENABLE
    )
    gpu_initialized = True

    report_devices(ctx)

def ensure_initialized():
    global gpu_initialized
    if not gpu_initialized:
        initialize_with_ctx(pyopencl.create_some_context())

def ptr(arr):
    if type(arr) is pyopencl.array.Array:
        return arr.data
    return arr

def to_gpu(arr, float_type):
    ensure_initialized()
    if type(arr) is pyopencl.array.Array:
        return arr
    to_type = arr.astype(float_type)
    return pyopencl.array.to_device(gpu_queue, to_type)

def zeros_gpu(shape, float_type):
    ensure_initialized()
    return pyopencl.array.zeros(gpu_queue, shape, float_type)

def empty_gpu(shape, float_type):
    ensure_initialized()
    return pyopencl.array.empty(gpu_queue, shape, float_type)

def threaded_get(arr):
    return arr.get()

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        kernel = getattr(self.module, name)
        def provide_queue_wrapper(*args, grid = None, block = None, **kwargs):
            global_size = [b * g for b, g in zip(grid, block)]
            arg_ptrs = [ptr(a) for a in args]
            return kernel(gpu_queue, global_size, block, *arg_ptrs, **kwargs)
        return provide_queue_wrapper

def compile(code):
    ensure_initialized()

    compile_options = []

    debug_opts = ['-g', '-Werror']
    # compile_options.extend(debug_opts)
    fast_opts = [
        # '-cl-finite-math-only',
        '-cl-unsafe-math-optimizations',
        # '-cl-no-signed-zeros',
        '-cl-mad-enable',
        # '-cl-strict-aliasing'
    ]
    compile_options.extend(fast_opts)

    return ModuleWrapper(pyopencl.Program(
        gpu_ctx, code
    ).build(options = compile_options))

cluda_preamble = """
// taken from pyopencl._cluda
#define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
// 'static' helps to avoid the "no previous prototype for function" warning
#if __OPENCL_VERSION__ >= 120
#define WITHIN_KERNEL static
#else
#define WITHIN_KERNEL
#endif
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_DYNAMIC __local
#define LOCAL_MEM_ARG __local
#define CONSTANT __constant
// INLINE is already defined in Beignet driver
#ifndef INLINE
#define INLINE inline
#endif
#define SIZE_T size_t
#define VSIZE_T size_t
// used to align fields in structures
#define ALIGN(bytes) __attribute__ ((aligned(bytes)))
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
"""
