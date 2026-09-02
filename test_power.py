import time
import jax
import jax.numpy as jnp
from jaxoplanet.experimental import calc_poly_coeffs
from models.core import get_I_power2 

@jax.jit
def compute_power2_full(c1, c2, mus):
    return 1 - c1 * (1 - jnp.power(mus, c2))
#def compute_power2_full(c1, c2, mus):
#    profile = get_I_power2(c1, c2, mus)
#    return calc_poly_coeffs(mus, profile, poly_degree=12)

def benchmark():
    c1 = jnp.array(0.5)
    c2 = jnp.array(0.2)
    mus = jnp.linspace(0.0, 1.0, 200)
    
    print("1. Compiling/Warming up...")
    _ = compute_power2_full(c1, c2, mus).block_until_ready()
    
    print("2. Benchmarking...")
    n_loops = 1000
    start = time.time()
    for _ in range(n_loops):
        _ = compute_power2_full(c1, c2, mus).block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / n_loops
    print(f"Average execution time over {n_loops} runs: {avg_time * 1e6:.2f} microseconds")

if __name__ == "__main__":
    benchmark()
