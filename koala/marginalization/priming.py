"""Prime exact sampler executables without running disposable Markov chains."""
from concurrent.futures import ThreadPoolExecutor
import time
import jax


def prime_laplace_runner(runner):
    """Overlap LLVM compilation of initialization and sampling programs.

    Trace both programs serially: NumPyro's effect-handler stack is shared.
    Only XLA compilation runs in the background. Dummy inputs are abstract
    shapes; every actual lane still runs its own MAP and fresh adaptation.
    """
    if runner.options.get('mass_matrix') != 'laplace' or runner.options.get('laplace_fuse_program'):
        raise ValueError('Priming requires an unfused Laplace sampler.')
    build = runner._build_programs
    runner._grid_priming_timings = []
    def build_and_prime(*args, **kwargs):
        build(*args, **kwargs)
        initialize = runner._init_program
        sample = runner._laplace_sample_program
        def prepare(*arguments):
            start = time.perf_counter()
            lowered_initialize = initialize.lower(*arguments)
            abstract_states = lowered_initialize.out_info[0]
            # Initializer: init, errors, observations, varying, keys, t, shared.
            sample_arguments = (abstract_states, arguments[5], arguments[1],
                                arguments[2], arguments[3], arguments[6])
            lowered_sample = sample.lower(*sample_arguments)
            trace_seconds = time.perf_counter()-start
            with ThreadPoolExecutor(max_workers=1) as compiler:
                future = compiler.submit(lowered_sample.compile)
                executable = lowered_initialize.compile()
                initialized = executable(*arguments)
                jax.block_until_ready(initialized)
                runner._laplace_sample_program = future.result()
            runner._init_program = executable
            runner._grid_priming_timings.append(dict(trace_seconds=trace_seconds,
                initialization_and_compile_seconds=time.perf_counter()-start,
                disposable_sampling_iterations=0))
            return initialized
        runner._init_program = prepare
    runner._build_programs = build_and_prime
    return runner
