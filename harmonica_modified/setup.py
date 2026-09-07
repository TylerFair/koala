import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent
OPENMP_INSTALL_NAME = None
OPENMP_RUNTIME_PATH = None
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext


def get_compile_args():
    args = ["-O3", "-DNDEBUG"]
    if os.environ.get("HARMONICA_MARCH_NATIVE", "").lower() in {"1", "true", "yes", "on"}:
        args.append("-march=native")
    extra = os.environ.get("HARMONICA_EXTRA_COMPILE_ARGS", "").strip()
    if extra:
        args.extend(extra.split())
    return args


def get_openmp_args():
    global OPENMP_INSTALL_NAME
    global OPENMP_RUNTIME_PATH

    enabled = os.environ.get("HARMONICA_ENABLE_OPENMP", "").lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        return [], []

    link_args = []
    if sys.platform == "darwin":
        compile_args = ["-Xpreprocessor", "-fopenmp", "-DEIGEN_DONT_PARALLELIZE"]
        lib_candidates = []

        lib_dir = os.environ.get("HARMONICA_OPENMP_LIB_DIR")
        if lib_dir:
            lib_candidates.extend([
                Path(lib_dir) / "libomp.dylib",
                Path(lib_dir) / "libgomp.dylib",
            ])

        lib_candidates.extend(Path(sys.prefix).rglob("libomp.dylib"))
        lib_candidates.extend(Path("/opt/homebrew").rglob("libomp.dylib"))
        lib_candidates.extend(Path("/usr/local").rglob("libomp.dylib"))
        lib_candidates.extend(Path(sys.prefix).rglob("libgomp.dylib"))
        lib_candidates.extend(Path("/opt/homebrew").rglob("libgomp.dylib"))
        lib_candidates.extend(Path("/usr/local").rglob("libgomp.dylib"))

        source_lib = None
        for candidate in lib_candidates:
            if candidate.is_file():
                source_lib = candidate
                break

        if source_lib is not None:
            OPENMP_RUNTIME_PATH = source_lib
            if source_lib.name == "libomp.dylib":
                try:
                    otool_output = subprocess.check_output(
                        ["otool", "-D", str(source_lib)],
                        text=True,
                    )
                    OPENMP_INSTALL_NAME = otool_output.splitlines()[-1].strip()
                except Exception:
                    OPENMP_INSTALL_NAME = str(source_lib)
            link_args.extend([str(source_lib), f"-Wl,-rpath,{source_lib.parent}"])
        else:
            link_args.append("-lomp")

        return compile_args, link_args
    if os.name == "nt":
        return ["/openmp", "-DEIGEN_DONT_PARALLELIZE"], []
    return ["-fopenmp", "-DEIGEN_DONT_PARALLELIZE"], ["-fopenmp"]


def get_cuda_args():
    enabled = os.environ.get("HARMONICA_ENABLE_CUDA", "").lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        return [], [], [], [], []

    cuda_home_candidates = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            cuda_home_candidates.append(Path(value))
    cuda_home_candidates.extend([Path("/usr/local/cuda"), Path("/opt/cuda")])

    include_dirs = []
    library_dirs = []
    for cuda_home in cuda_home_candidates:
        include_dir = cuda_home / "include"
        if (include_dir / "cuda_runtime_api.h").is_file():
            include_dirs.append(str(include_dir))
        for lib_dir in ("lib64", "lib"):
            candidate = cuda_home / lib_dir
            if candidate.is_dir():
                library_dirs.append(str(candidate))

    if not include_dirs:
        raise RuntimeError(
            "HARMONICA_ENABLE_CUDA is set but CUDA headers were not found."
        )
    if not library_dirs:
        raise RuntimeError(
            "HARMONICA_ENABLE_CUDA is set but CUDA libraries were not found."
        )

    define_macros = [("HARMONICA_ENABLE_CUDA", "1")]
    libraries = ["cudart"]
    extra_link_args = []
    if sys.platform != "darwin":
        extra_link_args.append(f"-Wl,-rpath,{library_dirs[0]}")
    return define_macros, include_dirs, library_dirs, libraries, extra_link_args

openmp_compile_args, openmp_link_args = get_openmp_args()
cuda_define_macros, cuda_include_dirs, cuda_library_dirs, cuda_libraries, cuda_link_args = get_cuda_args()


def _find_nvcc():
    """Locate the nvcc compiler, checking CUDA_HOME/CUDA_PATH then PATH."""
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            nvcc = Path(value) / "bin" / "nvcc"
            if nvcc.is_file():
                return str(nvcc)
    for prefix in (Path("/usr/local/cuda"), Path("/opt/cuda")):
        nvcc = prefix / "bin" / "nvcc"
        if nvcc.is_file():
            return str(nvcc)
    # Fall back to PATH
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    raise RuntimeError(
        "HARMONICA_ENABLE_CUDA is set but nvcc was not found. "
        "Set CUDA_HOME or ensure nvcc is on PATH."
    )


CUDA_KERNEL_SOURCES = [
    "harmonica/core/power2_nc1_kernel.cu",
]


def _compile_cuda_kernels(build_temp):
    """Compile .cu files with nvcc and return a list of .o paths."""
    nvcc = _find_nvcc()
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-arch=sm_70",
        "-Xcompiler", "-fPIC",
        "-c",
    ]
    # Add the same include directories the C++ extension uses.
    include_flags = []
    for inc in ["vendor/eigen", pybind11.get_include(), "harmonica"] + cuda_include_dirs:
        include_flags.extend(["-I", str(ROOT / inc) if not os.path.isabs(inc) else inc])

    obj_files = []
    for cu_src in CUDA_KERNEL_SOURCES:
        cu_path = ROOT / cu_src
        if not cu_path.is_file():
            raise FileNotFoundError(f"CUDA source not found: {cu_path}")

        obj_name = cu_path.stem + ".o"
        obj_dir = Path(build_temp) / cu_path.parent
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_path = obj_dir / obj_name

        cmd = [nvcc] + nvcc_flags + include_flags + ["-o", str(obj_path), str(cu_path)]
        print(f"[harmonica] nvcc: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        obj_files.append(str(obj_path))

    return obj_files


class build_ext_with_openmp(build_ext):
    def build_extension(self, ext):
        # --- CUDA kernel compilation step ---
        cuda_enabled = os.environ.get("HARMONICA_ENABLE_CUDA", "").lower() in {
            "1", "true", "yes", "on"
        }
        if cuda_enabled and ext.name == "harmonica.core.bindings":
            obj_files = _compile_cuda_kernels(self.build_temp)
            ext.extra_objects.extend(obj_files)

        super().build_extension(ext)

        # --- macOS OpenMP install-name fixup ---
        if not OPENMP_INSTALL_NAME or not OPENMP_RUNTIME_PATH:
            return
        if sys.platform != "darwin":
            return
        built_ext = Path(self.get_ext_fullpath(ext.name))
        subprocess.run(
            [
                "install_name_tool",
                "-change",
                OPENMP_INSTALL_NAME,
                "@rpath/libomp.dylib",
                str(built_ext),
            ],
            check=True,
        )


ext_modules = [
    Pybind11Extension(
        "harmonica.core.bindings",
        [
             'harmonica/orbit/kepler.cpp',
             'harmonica/orbit/trajectories.cpp',
             'harmonica/orbit/gradients.cpp',
             'harmonica/light_curve/fluxes.cpp',
             'harmonica/light_curve/gradients.cpp',
             'harmonica/core/bindings.cpp'
        ],
        include_dirs=[
            "vendor/eigen",
            pybind11.get_include(),
            "harmonica",
        ] + cuda_include_dirs,
        language="c++",
        extra_compile_args=get_compile_args() + openmp_compile_args,
        extra_link_args=openmp_link_args + cuda_link_args,
        define_macros=cuda_define_macros,
        library_dirs=cuda_library_dirs,
        libraries=cuda_libraries,
    ),
]

setup(
    name="planet-harmonica",
    version="0.2.1",
    author="David Grant",
    author_email="david.grant@bristol.ac.uk",
    url="https://github.com/DavoGrant/harmonica",
    license="MIT",
    license_files=["LICENSE", "vendor/eigen/COPYING*"],
    packages=find_packages(where="."),
    include_package_data=True,
    description="Light curves for exoplanet transmission mapping.",
    long_description="Light curves for exoplanet transmission mapping.",
    python_requires=">=3.6",
    install_requires=["numpy", "jax>=0.5.3", "jaxlib>=0.5.3"],
    cmdclass={"build_ext": build_ext_with_openmp},
    ext_modules=ext_modules,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
