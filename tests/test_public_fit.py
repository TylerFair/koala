"""The notebook entry point prepares devices before loading the pipeline."""
import subprocess
import sys
import textwrap


def test_fit_prepares_devices_and_forwards_config(tmp_path):
    config = tmp_path / 'fit.yaml'
    config.write_text('host_device: cpu\nflags:\n  cpu_host_devices: 3\n')
    script = textwrap.dedent('''
        import os
        import sys
        from pathlib import Path
        from types import ModuleType
        import koala
        assert 'jax' not in sys.modules
        import builtins
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name in ('config', 'pipeline'):
                assert '--xla_force_host_platform_device_count=3' in os.environ['XLA_FLAGS']
            return original_import(name, *args, **kwargs)
        builtins.__import__ = checked_import
        pipeline = ModuleType('koala.pipeline')
        seen_verbose = []
        def run(cfg, config_path=None, verbose=False):
            seen_verbose.append(verbose)
            assert cfg['host_device'] == 'cpu'
            assert 'verbose' not in cfg
            assert isinstance(verbose, bool)
            assert config_path == sys.argv[1]
            return 'pipeline result'
        pipeline.run = run
        sys.modules['koala.pipeline'] = pipeline
        assert koala.fit(Path(sys.argv[1])) == 'pipeline result'
        assert koala.fit(Path(sys.argv[1]), verbose=True) == 'pipeline result'
        assert seen_verbose == [False, True]
    ''')
    import os
    env = dict(os.environ)
    env.pop('XLA_FLAGS', None)
    env.pop('KOALA_CPU_HOST_DEVICES', None)
    subprocess.run([sys.executable, '-c', script, str(config)], env=env, check=True)
