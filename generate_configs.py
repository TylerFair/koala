import os
import re
from pathlib import Path
import ast
import yaml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
'''
  spot_center: 59910.405
  spot_center2: 59910.428
  spot_width: 0.005
  spot_width2: 0.005
  spot_amp: 0.001
  spot_amp2: 0.001
'''
class JWSTConfigGenerator:
    def __init__(self, fits_dir, output_dir, archive_csv, reduction_list_csv, transit_times_file=None, base_path='/scratch/midway3/tfairnington/'):
        self.fits_dir = Path(fits_dir)
        self.output_dir = Path(output_dir)
        self.base_path = base_path
        self.archive_csv = archive_csv
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the exoplanet archive CSV
        print(f"Loading archive from: {archive_csv}")
        self.archive_df = pd.read_csv(archive_csv, comment='#')
        print(f"Loaded {len(self.archive_df)} rows from archive")
        
        # Load the reduction list CSV to map star names to planet names
        print(f"Loading reduction list from: {reduction_list_csv}")
        self.reduction_df = pd.read_csv(reduction_list_csv)
        
        # Load per-dataset configuration notes (t0, detrending, masks, etc.)
        # This replaces the old transit_times.txt.
        self.config_notes = {}
        self.transit_times = {}
        if transit_times_file and os.path.exists(transit_times_file):
            print(f"Loading config notes from: {transit_times_file}")
            # Parse lines like:
            #   <fname> <t0> [trend] [trendparams] [timemasks]
            # Where only <fname> and <t0> are required; missing fields default to 'none'/None.
            with open(transit_times_file, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 4)
                    if len(parts) < 2:
                        raise ValueError(f"Could not parse config_notes line (need at least fname and t0): {raw}")
                    fname = parts[0]
                    t0_str = parts[1]
                    trend = parts[2] if len(parts) >= 3 else "none"
                    trendparams_raw = parts[3] if len(parts) >= 4 else None
                    timemasks_raw  = parts[4] if len(parts) >= 5 else None
                    d = {"fname": fname, "t0": t0_str, "trend": trend,
                        "detrending_type": trend, "trendparams": trendparams_raw, "timemasks": timemasks_raw}

                    def _parse_maybe(v):
                        if v is None:
                            return None
                        v = v.strip()
                        if v.lower() == "none":
                            return None
                        try:
                            return ast.literal_eval(v)
                        except Exception:
                            return v

                    key = d["fname"].strip().upper()
                    t0 = float(d["t0"])
                    trend = d["trend"].strip()
                    trendparams = _parse_maybe(d.get("trendparams"))
                    timemasks = _parse_maybe(d.get("timemasks"))

                    self.config_notes[key] = {
                        "t0": t0,
                        "trend": trend,
                        "detrending_type": trend,
                        "trendparams": trendparams,
                        "timemasks": timemasks,
                    }
                    self.transit_times[key] = t0  # keep legacy access pattern
            print(f"Loaded {len(self.config_notes)} config note rows")
        # Create mapping from star name to planet name
        self.star_to_planet = {}
        for _, row in self.reduction_df.iterrows():
            star_name = str(row['hostname_nn']).strip()
            letter = str(row['letter_nn']).strip()
            star_name_key = star_name.upper().replace('-', '').replace(' ', '')
            planet_name = f"{star_name} {letter}".upper()
            self.star_to_planet[star_name_key] = planet_name
            
            star_parts = star_name.upper().split()
            if len(star_parts) > 1 and star_parts[-1] in ['A', 'B', 'C']:
                base_star = ' '.join(star_parts[:-1])
                base_star_key = base_star.replace('-', '').replace(' ', '')
                if base_star_key not in self.star_to_planet:
                    self.star_to_planet[base_star_key] = planet_name
        
        print(f"Loaded {len(self.star_to_planet)} star-to-planet mappings")
        
        # Instrument/grating to order mapping
        self.instrument_config = {
            'G395H': {'order': None, 'nrs_depends_on_filename': True},
            'G395M': {'order': 1, 'nrs': 1},
            'G140H': {'order': None, 'nrs_depends_on_filename': True},
            'PRISM': {'order': None, 'nrs': 1},
            'SOSS': {'order': 1, 'nrs': None}  # Default, can be overridden
        }
        # Define Phase Curve targets
        self.phase_curve_rules = self._define_phase_curve_rules()
        
        # Define wavelength limit rules for specific datasets
        self.wavelength_rules = self._define_wavelength_rules()
    def _define_phase_curve_rules(self):
        """
        Define which targets are Phase Curves.
        You can identify them by Star Name only, or (Star Name, Visit).
        """
        phase_curves = set()
        
        phase_curves.add('WASP-121')
        phase_curves.add('NGTS-10')
        phase_curves.add('LTT-9779')
        phase_curves.add('LTT9779')
        phase_curves.add('LTT 9779')
        return phase_curves
    
    def _define_wavelength_rules(self):
        """Define wavelength limit rules for specific star/instrument/visit combinations."""
        rules = {}
        rules[('WASP-39', 'PRISM', 1, None)] = {'wl_min': 2.0, 'wl_max': 5.0}
        rules[('HATS-72', 'PRISM', 1, None)] = {'wl_min': 2.0, 'wl_max': 5.0} 
        return rules
    
    
    def get_masking_for_config(self, star_name, grating, nrs, visit):
        """Return time-masking instructions for this dataset.

        Priority:
          1) If `timemasks` is provided in config_notes.txt for this dataset, use it.
          2) Otherwise, if this target is marked as a phase curve, use the special
             sentinel mask value `cut_phase_to_transit` (the downstream fitter
             interprets this to cut phase-curve baselines to the transit window).
          3) Otherwise, return None (no masking).

        Accepted `timemasks` formats (parsed via ast.literal_eval):
          - [start, end]
          - [[start1, end1], [start2, end2], ...]
          - {'mask_start': start, 'mask_end': end}
          - a list of dicts [{'mask_start':..., 'mask_end':...}, ...]
        """
        note = self.get_note_for_config(star_name, grating, nrs, visit)

        # --- 1) Explicit timemasks from config_notes.txt ---
        if note is not None and note.get("timemasks") is not None:
            tm = note["timemasks"]

            # Already-dict
            if isinstance(tm, dict):
                return tm

            # List handling
            if isinstance(tm, list):
                # [start, end]
                if len(tm) == 2 and all(not isinstance(x, (list, dict)) for x in tm):
                    return {"mask_start": tm[0], "mask_end": tm[1]}

                # [[start,end], ...]  -> list-of-dicts
                if all(isinstance(x, list) and len(x) == 2 for x in tm):
                    return [{"mask_start": x[0], "mask_end": x[1]} for x in tm]

                # list-of-dicts passthrough
                if all(isinstance(x, dict) for x in tm):
                    return tm

            raise ValueError(
                f"Unrecognized timemasks format for {star_name} {grating} {nrs} {visit}: {tm}"
            )

        # --- 2) Phase-curve default masking (original behavior) ---
        is_phase_curve = False
        if star_name in self.phase_curve_rules:
            is_phase_curve = True
        elif (star_name, visit) in self.phase_curve_rules:
            is_phase_curve = True

        if is_phase_curve:
            return {"mask_start": "cut_phase_to_transit", "mask_end": "cut_phase_to_transit"}

        # --- 3) No masking ---
        return None

        tm = note["timemasks"]
        # Accepted formats:
        #  - [start, end]
        #  - [[start1, end1], [start2, end2], ...]
        #  - {"mask_start": ..., "mask_end": ...}
        if isinstance(tm, dict):
            return tm
        if isinstance(tm, (list, tuple)) and len(tm) == 2 and all(isinstance(x, (int, float)) or x is None for x in tm):
            return {"mask_start": tm[0], "mask_end": tm[1]}
        if isinstance(tm, (list, tuple)) and len(tm) > 0 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in tm):
            return {"mask_start": [x[0] for x in tm], "mask_end": [x[1] for x in tm]}
        raise ValueError(f"Unrecognized timemasks format for {star_name} {grating}: {tm}")

        
    def get_wavelength_limits_for_config(self, star_name, grating, nrs, visit):
        key = (star_name, grating, nrs, visit)
        if key in self.wavelength_rules:
            return self.wavelength_rules[key]
        key_no_visit = (star_name, grating, nrs, None)
        if key_no_visit in self.wavelength_rules:
            return self.wavelength_rules[key_no_visit]
        return None
    
    
    def get_detrending_for_config(self, star_name, grating, nrs, visit):
        """Return detrending settings for this dataset from config_notes.txt.

        If no entry exists (or detrending is None/none), returns None.
        """
        note = self.get_note_for_config(star_name, grating, nrs, visit)
        if note is None:
            return None
        detrending_type = note.get("detrending_type")
        if detrending_type is None or str(detrending_type).lower() in ("none", "null"):
            return None
        return {
            "detrending_type": str(detrending_type).strip(),
            "trendparams": note.get("trendparams"),
        }
        
    def parse_filename(self, filename):
        base = filename.replace('.fits', '')
        star_match = re.match(r'^([A-Za-z0-9\-]+)', base)
        star_name = star_match.group(1) if star_match else None
        nrs_match = re.search(r'nrs([12])', base)
        nrs = int(nrs_match.group(1)) if nrs_match else None
        grating = None
        for g in ['G395H', 'G395M', 'G140H', 'PRISM']:
            if g in base:
                grating = g
                break
        if grating is None and nrs is None:
            grating = 'SOSS'
        visit = None
        visit_match = re.search(r'_V(\d+)', base, re.IGNORECASE)
        if visit_match:
            visit = int(visit_match.group(1))
        else:
            prism_match = re.search(r'PRISM(\d+)', base)
            if prism_match:
                visit = int(prism_match.group(1))
        return star_name, nrs, grating, visit
    

    def _keys_to_try(self, star_name, nrs, grating, visit):
        keys_to_try = []
        if grating == 'SOSS':
            if visit:
                keys_to_try.append(f"{star_name}_NIRISS_V{visit}")
            keys_to_try.append(f"{star_name}_NIRISS")
        else:
            # --- try NRS-first (existing) ---
            if nrs and visit:
                keys_to_try.append(f"{star_name}_NRS{nrs}_{grating}_V{visit}")
            if nrs:
                keys_to_try.append(f"{star_name}_NRS{nrs}_{grating}")

            # --- ALSO try grating-first (your output_dir style) ---
            if nrs and visit:
                keys_to_try.append(f"{star_name}_{grating}_NRS{nrs}_V{visit}")
            if nrs:
                keys_to_try.append(f"{star_name}_{grating}_NRS{nrs}")

            # fallbacks
            if visit:
                keys_to_try.append(f"{star_name}_{grating}_V{visit}")
            keys_to_try.append(f"{star_name}_{grating}")
        print("[DEBUG] keys_to_try:")
        for k in keys_to_try:
            print("   ", k)


        return [k.upper() for k in keys_to_try]


    def _get_config_note(self, star_name, nrs, grating, visit):
        print("\n[DEBUG] _get_config_note called with:")
        print(f"  star_name = {star_name}")
        print(f"  grating   = {grating}")
        print(f"  nrs       = {nrs}")
        print(f"  visit     = {visit}")

        for key in self._keys_to_try(star_name, nrs, grating, visit):
            if key in self.config_notes:
                return key, self.config_notes[key]
        return None, None

    # Back-compat / readability alias
    def get_note_for_config(self, star_name, grating, nrs, visit):
        """Return the parsed config_notes entry (or None) for this dataset.

        Parameters follow the common call pattern elsewhere in the script:
        (star_name, grating, nrs, visit). Internally we reuse _get_config_note().
        """
        _, note = self._get_config_note(star_name, nrs, grating, visit)
        return note


    def get_transit_time(self, star_name, nrs, grating, visit):
        for key in self._keys_to_try(star_name, nrs, grating, visit):
            if key in self.transit_times:
                return self.transit_times[key]
        return 0.0
    
    def query_archive_csv(self, star_name):
        star_name_key = star_name.upper().replace('-', '').replace(' ', '')
        if star_name_key not in self.star_to_planet:
            print(f"  ❌ ERROR: No mapping found for star {star_name}")
            print(f"     Add {star_name} to reduction_list.csv")
            raise KeyError(f"Star name {star_name} not found in reduction_list.csv")
        planet_name = self.star_to_planet[star_name_key]
        print(f"  🔍 Mapped {star_name} → {planet_name}")
        planet_data = self.archive_df[self.archive_df['pl_name'].str.upper() == planet_name].copy()
        if len(planet_data) == 0:
            print(f"  ❌ ERROR: No data found for {planet_name} in archive")
            raise ValueError(f"Planet {planet_name} not found in archive")
        planet_data['releasedate'] = pd.to_datetime(planet_data['releasedate'])
        planet_data = planet_data.sort_values('releasedate', ascending=False)
        def get_param(param_name, default=None):
            for _, row in planet_data.iterrows():
                val = row.get(param_name)
                if pd.notna(val) and val != '':
                    return float(val)
            return default
        params = {
            'period': get_param('pl_orbper'),
            'duration': get_param('pl_trandur'),
            'b': get_param('pl_imppar', 0.5),
            'rprs': get_param('pl_ratror'),
            'feh': get_param('st_met', 0.0),
            'teff': get_param('st_teff', 5700),
            'logg': get_param('st_logg', 4.4),
        }
        if params['duration'] is not None:
            params['duration'] = params['duration'] / 24.0
        return params
    
    def create_config(self, filename, planet_params, star_name, force_order=None):
        _, nrs, grating, visit = self.parse_filename(filename)
        if grating is None:
            print(f"  ⚠️  Could not determine grating for {filename}")
            return None
        t0 = self.get_transit_time(star_name, nrs, grating, visit)
        if t0 == 0.0:
            print(f"  ⚠️  No transit time found - using 0.0 as placeholder")
        else:
            print(f"  ✓ Found transit time: {t0}")
            
        masking = self.get_masking_for_config(star_name, grating, nrs, visit)
        if masking:
            print(f"  🎯 Applying masking rules for {star_name}")
            
        wavelength_limits = self.get_wavelength_limits_for_config(star_name, grating, nrs, visit)
        if wavelength_limits:
            print(f"  📏 Applying wavelength limits for {star_name}")
            
        detrending_config = self.get_detrending_for_config(star_name, grating, nrs, visit)
        if detrending_config:
            print(f"  📊 Applying custom detrending for {star_name}")
            
        if grating == 'SOSS':
            instrument = 'NIRISS/SOSS'
            # If force_order is provided, use it, otherwise default to 1
            order = force_order if force_order is not None else 1
            nrs_val = None
        else:
            instrument = f'NIRSPEC/{grating}'
            config = self.instrument_config.get(grating, {})
            order = config.get('order')
            if config.get('nrs_depends_on_filename', False):
                nrs_val = nrs
            else:
                nrs_val = config.get('nrs')
        
        detrending_type_for_dir = 'TREND'
        if detrending_config and 'detrending_type' in detrending_config:
            detrend = str(detrending_config['detrending_type']).strip()
            # Folder-safe label
            detrending_type_for_dir = detrend.upper()
            detrending_type_for_dir = detrending_type_for_dir.replace('-', '_').replace(' ', '_')
        # DETERMINE OUTPUT DIRECTORY NAME
        if nrs_val:
            if visit:
                output_dir_name = f'{star_name}_{grating}_NRS{nrs_val}_V{visit}_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
            else:
                output_dir_name = f'{star_name}_{grating}_NRS{nrs_val}_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
        else:
            # Handle SOSS / NIRISS
            if grating == 'SOSS':
                # Insert order into directory name for SOSS
                soss_suffix = f"ORDER{order}"
                if visit:
                    output_dir_name = f'{star_name}_SOSS_{soss_suffix}_V{visit}_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
                else:
                    output_dir_name = f'{star_name}_SOSS_{soss_suffix}_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
            else:
                if visit:
                    output_dir_name = f'{star_name}_SOSS_V{visit}_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
                else:
                    output_dir_name = f'{star_name}_SOSS_GAUSSIANLD_POWER2_{detrending_type_for_dir}'
        
        if grating == 'PRISM':
            resolution_config = {'high': 'native', 'low': 20}
        else:
            resolution_config = {'high': 'reference', 'low': 20, 'reference_grid': 'prism_template.csv'}
        
        detrending_type = 'linear'
        trendparams = None
        if detrending_config:
            detrending_type = str(detrending_config.get('detrending_type', detrending_type)).strip()
            trendparams = detrending_config.get('trendparams', None)

        flags = {
            'detrending_type': detrending_type,
            'interpolate_trend': False,
            'interpolate_ld': False,
            'fix_ld': False,
            'need_lowres': True,
            'mask_integrations_start': None,
            'ld_profile': 'power2'  # --- NEW ADDITION ---
        }
        
        if masking:
            # Case 1: Multiple masks (List of dictionaries)
            if isinstance(masking, list):
                # Extract all starts and ends into simple lists for the YAML writer
                flags['mask_start'] = [m['mask_start'] for m in masking]
                flags['mask_end']   = [m['mask_end'] for m in masking]
            
            # Case 2: Single mask (Dictionary)
            elif isinstance(masking, dict):
                flags['mask_start'] = masking.get('mask_start')
                flags['mask_end']   = masking.get('mask_end') 

        # ---- Trend-parameter initial guesses from config_notes.txt ----
        dt = detrending_type.lower()
        if dt == 'linear_discontinuity':
            # Expect trendparams like [t_jump]
            if trendparams is None or (isinstance(trendparams, (list, tuple)) and len(trendparams) < 1):
                raise ValueError(f"linear_discontinuity requires trendparams like [t_jump] for {star_name} {grating}")
            t_jump = trendparams[0] if isinstance(trendparams, (list, tuple)) else trendparams
            flags['t_jump_guess'] = float(t_jump)
            flags['jump_guess'] = -0.0005
        elif dt == 'spot':
            if trendparams is None or (isinstance(trendparams, (list, tuple)) and len(trendparams) < 1):
                raise ValueError(f"spot requires trendparams like [spot_center] for {star_name} {grating}")
            c1 = trendparams[0] if isinstance(trendparams, (list, tuple)) else trendparams
            flags['fix_ld'] = False
            flags['ld_profile'] = 'power2'
            flags['spot_center'] = float(c1)
            flags['spot_width'] = 0.005
            flags['spot_amp'] = 0.001
        elif dt == '2spot':
            if trendparams is None or not (isinstance(trendparams, (list, tuple)) and len(trendparams) >= 2):
                raise ValueError(f"2spot requires trendparams like [spot_center1, spot_center2] for {star_name} {grating}")
            flags['fix_ld'] = False
            flags['ld_profile'] = 'power2'
            flags['spot_center'] = float(trendparams[0])
            flags['spot_center2'] = float(trendparams[1])
            flags['spot_width'] = 0.005
            flags['spot_width2'] = 0.005
            flags['spot_amp'] = 0.001
            flags['spot_amp2'] = 0.001

        config = {
            'planet': {
                'name': star_name,
                'period': planet_params.get('period', 3.0),
                'duration': planet_params.get('duration', 0.1),
                't0': t0,
                'b': planet_params.get('b', 0.5),
                'rprs': planet_params.get('rprs', 0.1),
            },
            'stellar': {
                'feh': planet_params.get('feh', 0.0),
                'teff': planet_params.get('teff', 5700),
                'logg': planet_params.get('logg', 4.4),
                'ld_model': 'stagger',
                'ld_data_path': '../exotic_ld_data',
            },
            'instrument': instrument,
            'path': self.base_path,
            'input_dir': 'FITS',
            'output_dir': output_dir_name,
            'fits_file': filename,
            'resolution': resolution_config,
            'flags': flags,
            'outlier_clip': {'whitelight_sigma': 4, 'spectroscopic_sigma': 4},
            'host_device': 'gpu',
        }
        if grating == 'PRISM':
            config['time_binning'] = {
                'enabled': True,
                'dt_seconds': 20,
                'method': 'mean',
                'whitelight': True,
                'spectroscopic': False
            }

        if wavelength_limits:
            if 'wl_min' in wavelength_limits:
                config['wl_min'] = wavelength_limits['wl_min']
            if 'wl_max' in wavelength_limits:
                config['wl_max'] = wavelength_limits['wl_max']
        if order is not None:
            config['order'] = order
        if nrs_val is not None:
            config['nrs'] = nrs_val
        
        return config, visit
    
    def save_config(self, config, star_name, nrs, grating, visit=None):
        if nrs:
            if visit:
                filename = f"{star_name}_nrs{nrs}_{grating.lower()}_v{visit}_config.yaml"
            else:
                filename = f"{star_name}_nrs{nrs}_{grating.lower()}_config.yaml"
        else:
            # Modified for SOSS to include order in filename
            order_suffix = ""
            if 'order' in config and config['order'] is not None:
                order_suffix = f"_order{config['order']}"
            
            if visit:
                filename = f"{star_name}_soss{order_suffix}_v{visit}_config.yaml"
            else:
                filename = f"{star_name}_soss{order_suffix}_config.yaml"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"# Auto-generated config for {star_name}")
            if visit:
                f.write(f" Visit {visit}")
            f.write("\n")
            t0_val = config['planet']['t0']
            if t0_val == 0.0:
                f.write(f"# ⚠️  WARNING: No transit time found! Update t0 parameter manually!\n\n")
            else:
                f.write(f"# Transit time (t0) automatically assigned: {t0_val}\n\n")
            self._write_yaml_custom(f, config)
        return filepath
    
    def _write_yaml_custom(self, f, config):
        f.write("planet:\n")
        f.write(f"  name: {config['planet']['name']}\n")
        f.write(f"  period: {config['planet']['period']}\n")
        f.write(f"  duration: {config['planet']['duration']}\n")
        f.write(f"  t0: {config['planet']['t0']}\n")
        f.write(f"  b: {config['planet']['b']}\n")
        f.write(f"  rprs: {config['planet']['rprs']}\n\n")
        
        f.write("stellar:\n")
        f.write(f"  feh: {config['stellar']['feh']:.2f}\n")
        f.write(f"  teff: {config['stellar']['teff']}\n")
        f.write(f"  logg: {config['stellar']['logg']:.2f}\n")
        f.write(f"  ld_model: \"{config['stellar']['ld_model']}\"\n")
        f.write(f"  ld_data_path: \"{config['stellar']['ld_data_path']}\"\n\n")
        
        f.write(f"instrument: '{config['instrument']}'\n")
        if 'order' in config and config['order'] is not None:
            f.write(f"order: {config['order']}\n")
        else:
            f.write("order: #\n")
        if 'nrs' in config and config['nrs'] is not None:
            f.write(f"nrs: {config['nrs']}\n")
            
        if 'wl_min' in config or 'wl_max' in config:
            f.write("wavelength_filter:\n")
            if 'wl_min' in config:
                f.write(f"  wl_min: {config['wl_min']}\n")
            if 'wl_max' in config:
                f.write(f"  wl_max: {config['wl_max']}\n")
        
        f.write(f"path: '{config['path']}'\n")
        f.write(f"input_dir: '{config['input_dir']}'\n")
        f.write(f"output_dir: '{config['output_dir']}'\n")
        f.write(f"fits_file: '{config['fits_file']}'\n")
        
        f.write("resolution:\n")
        if config['resolution']:
            for key in ("high", "low"):
                val = config['resolution'][key]
                f.write(f"  {key}: '{val}'\n" if isinstance(val, str) else f"  {key}: {val}\n")
        if 'reference_grid' in config['resolution']:
            f.write(f"  reference_grid: '{config['resolution']['reference_grid']}'\n")
        
        f.write("flags:\n")
        f.write(f"  detrending_type: '{config['flags']['detrending_type']}'\n")
        f.write(f"  interpolate_trend: {str(config['flags']['interpolate_trend'])}\n")
        f.write(f"  interpolate_ld: {str(config['flags']['interpolate_ld'])}\n")
        f.write(f"  fix_ld: {str(config['flags']['fix_ld'])}\n")
        f.write(f"  need_lowres: {str(config['flags']['need_lowres'])}\n")
        # --- NEW ADDITION ---
        if 'ld_profile' in config['flags']:
            f.write(f"  ld_profile: '{config['flags']['ld_profile']}'\n")

        # Optional trend-parameter initial guesses
        if 't_jump_guess' in config['flags']:
            f.write(f"  t_jump_guess: {config['flags']['t_jump_guess']}\n")
        if 'jump_guess' in config['flags']:
            f.write(f"  jump_guess: {config['flags']['jump_guess']}\n")

        if 'spot_center' in config['flags']:
            f.write(f"  spot_center: {config['flags']['spot_center']}\n")
        if 'spot_center2' in config['flags']:
            f.write(f"  spot_center2: {config['flags']['spot_center2']}\n")
        if 'spot_width' in config['flags']:
            f.write(f"  spot_width: {config['flags']['spot_width']}\n")
        if 'spot_width2' in config['flags']:
            f.write(f"  spot_width2: {config['flags']['spot_width2']}\n")
        if 'spot_amp' in config['flags']:
            f.write(f"  spot_amp: {config['flags']['spot_amp']}\n")
        if 'spot_amp2' in config['flags']:
            f.write(f"  spot_amp2: {config['flags']['spot_amp2']}\n")

        if 'mask_start' in config['flags']:

            mask_start = config['flags']['mask_start']
            if isinstance(mask_start, list):
                f.write(f"  mask_start: {mask_start}\n")
            else:
                f.write(f"  mask_start: {mask_start}\n")
        else:
            f.write(f"  #mask_start:\n")
        
        if 'mask_end' in config['flags']:
            mask_end = config['flags']['mask_end']
            if isinstance(mask_end, list):
                f.write(f"  mask_end: {mask_end}\n")
            else:
                f.write(f"  mask_end: {mask_end}\n")
        else:
            f.write(f"  #mask_end:\n")
            
        f.write(f"  mask_integrations_start: {config['flags']['mask_integrations_start']}\n")
        f.write(f"  #mask_integrations_end:\n")
       
        if 'time_binning' in config:
            f.write("time_binning:\n")
            f.write(f"  enabled: {str(config['time_binning']['enabled'])}\n")
            f.write(f"  dt_seconds: {config['time_binning']['dt_seconds']}\n")
            f.write(f"  method: '{config['time_binning']['method']}'\n\n")
            f.write(f"  whitelight: '{config['time_binning']['whitelight']}'\n\n")
            f.write(f"  spectroscopic: '{config['time_binning']['spectroscopic']}'\n\n")

        f.write("outlier_clip:\n")
        f.write(f"  whitelight_sigma: {config['outlier_clip']['whitelight_sigma']}\n")
        f.write(f"  spectroscopic_sigma: {config['outlier_clip']['spectroscopic_sigma']}\n\n")
        f.write(f"host_device: \"{config['host_device']}\"\n")
    
    def generate_all_configs(self):
        fits_files = sorted([f for f in os.listdir(self.fits_dir) if f.endswith('.fits')])
        print(f"Found {len(fits_files)} FITS files")
        print("=" * 60)
        planet_cache = {}
        for fits_file in fits_files:
            print(f"\n🔎 Processing: {fits_file}")
            star_name, nrs, grating, visit = self.parse_filename(fits_file)
            if star_name is None:
                print(f"  ⚠️  Could not parse star name, skipping")
                continue
            if star_name not in planet_cache:
                print(f"  🔍 Querying archive for {star_name}...")
                params = self.query_archive_csv(star_name)
                print(f"  ✓ Found parameters")
                planet_cache[star_name] = params
            else:
                print(f"  ✓ Using cached parameters")
                params = planet_cache[star_name]
            
            # --- MODIFIED LOOP FOR SOSS ORDERS ---
            if grating == 'SOSS':
                # Run twice, once for Order 1, once for Order 2
                orders_to_run = [1, 2]
                print(f"  ℹ️  SOSS detected: Generating configs for Order 1 and Order 2")
            else:
                # Run once with default order behavior
                orders_to_run = [None]
                
            for force_order in orders_to_run:
                result = self.create_config(fits_file, params, star_name, force_order=force_order)
                if result:
                    config, visit_num = result
                    filepath = self.save_config(config, star_name, nrs, grating, visit_num)
                    print(f"  ✅ Saved: {filepath.name}")

if __name__ == "__main__":
    fits_directory = "/scratch/midway3/tfairnington/FITS"
    output_directory = "./configs_o2"
    archive_csv = "/scratch/midway3/tfairnington/exoarchive_21nov2025.csv"
    reduction_list_csv = "/scratch/midway3/tfairnington/reduction_list.csv"
    transit_times_file = "/scratch/midway3/tfairnington/config_notes.txt"
    
    generator = JWSTConfigGenerator(
        fits_dir=fits_directory,
        output_dir=output_directory,
        archive_csv=archive_csv,
        reduction_list_csv=reduction_list_csv,
        transit_times_file=transit_times_file,
        base_path='/scratch/midway3/tfairnington/'
    )
    
    generator.generate_all_configs()
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)
