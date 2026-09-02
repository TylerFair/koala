import os
import re
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
        
        # Load transit times if provided
        self.transit_times = {}
        if transit_times_file and os.path.exists(transit_times_file):
            print(f"Loading transit times from: {transit_times_file}")
            with open(transit_times_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove backslashes and split
                        line = line.replace('\\', '').strip()
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].strip()
                            t0 = float(parts[1].strip())
                            self.transit_times[key.upper()] = t0
            print(f"Loaded {len(self.transit_times)} transit times")
        
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
        
        # Define masking rules for specific datasets
        self.masking_rules = self._define_masking_rules()
        
        # Define Phase Curve targets
        self.phase_curve_rules = self._define_phase_curve_rules()
        
        # Define wavelength limit rules for specific datasets
        self.wavelength_rules = self._define_wavelength_rules()
        
        # Define detrending type rules for specific datasets
        self.detrending_rules = self._define_detrending_rules()
    
    def _define_masking_rules(self):
        """Define masking rules for specific star/instrument/visit combinations."""
        rules = {}
        
        # WASP-39 PRISM NRS1
        rules[('WASP-39', 'PRISM', 1, None)] = {
            'mask_start': 59770.974816,
            'mask_end': 59770.9751014
        }
        # WASP-19 PRISM NRS1
        rules[('WASP-19', 'PRISM', 1, None)] = {
            'mask_start': 60756.661,
            'mask_end': 60756.6612951
        }
        # TOI-4010 NRS1 PRISM (mask from time onwards means no end)
        rules[('TOI-4010', 'PRISM', 1, None)] = {
            'mask_start': 60588.8785422,
            'mask_end': None
        }
        # TOI-3757 NRS1 PRISM V2
        rules[('TOI-3757', 'PRISM', 1, 2)] = {
            'mask_start': 60392.592477,
            'mask_end': 60392.592890
        }
        # TOI-3757 NRS1 PRISM V1 (multiple ranges)
        rules[('TOI-3757', 'PRISM', 1, 1)] = {
            'mask_start': [60389.2161031, 60389.227],
            'mask_end': [60389.2163966, 60389.23188]
        }
        # LTT-9779 NRS2 G395H (multiple ranges)
        rules[('LTT-9779', 'G395H', 2, None)] = {
            'mask_start': [60600.538267, 60600.799399],
            'mask_end': [60600.5390246, 60600.799812]
        }
        # LTT-9779 NRS1 G395H (multiple ranges)
        rules[('LTT-9779', 'G395H', 1, None)] = {
            'mask_start': [60600.538267, 60600.799399],
            'mask_end': [60600.5390246, 60600.799812]
        }
        # Kepler-12 NRS1 PRISM V2
        rules[('Kepler-12', 'PRISM', 1, 2)] = {
            'mask_start': 60422.3097284,
            'mask_end': 60422.3099674
        }
        
        # --- NEW ADDITION ---
        # HAT-P-18 SOSS (Applies to both orders, nrs is None for SOSS)
        rules[('HAT-P-18', 'SOSS', None, None)] = {
            'mask_start': 'jnp.min(t) + 0.152611',
            'mask_end': 'jnp.min(t) + 0.1716875'
        }
        
        return rules

    def _define_phase_curve_rules(self):
        """
        Define which targets are Phase Curves.
        You can identify them by Star Name only, or (Star Name, Visit).
        """
        phase_curves = set()
        
        phase_curves.add('WASP-121')
        phase_curves.add('NGTS-10')
        phase_curves.add('LTT-9779')

        return phase_curves
    
    def _define_detrending_rules(self):
        """Define detrending type rules for specific star/instrument/visit combinations."""
        rules = {}
        # WASP-39 G395H NRS1 and NRS2
        rules[('WASP-39', 'G395H', 1, None)] = {'detrending_type': 'linear_discontinuity'}
        rules[('WASP-39', 'G395H', 2, None)] = {'detrending_type': 'linear_discontinuity'}
        return rules
    
    def _define_wavelength_rules(self):
        """Define wavelength limit rules for specific star/instrument/visit combinations."""
        rules = {}
        rules[('WASP-39', 'PRISM', 1, None)] = {'wl_min': 2.0, 'wl_max': 5.0}
        rules[('HATS-72', 'PRISM', 1, None)] = {'wl_min': 2.0, 'wl_max': 5.0} 
        return rules
    
    def get_masking_for_config(self, star_name, grating, nrs, visit):
        """Get masking parameters for a specific configuration."""
        
        # 1. CHECK FOR HARDCODED MANUAL RULES FIRST (Existing logic)
        key = (star_name, grating, nrs, visit)
        existing_mask = None
        if key in self.masking_rules:
            existing_mask = self.masking_rules[key]
        else:
            # Try without visit if not found
            key_no_visit = (star_name, grating, nrs, None)
            if key_no_visit in self.masking_rules:
                existing_mask = self.masking_rules[key_no_visit]
        
        # 2. DEFAULT MASKS
        # Default Transits: mask first 30 minutes
        default_mask_start = None #'jnp.min(t)'
        default_mask_end = None #'jnp.min(t) + 0.020833'

        # Default Phase Curves: "duration_end"
        phase_curve_mask_val = 'cut_phase_to_transit'

        # 3. CHECK IF THIS IS A PHASE CURVE
        # Normalize star name for consistent lookup (optional, but good practice if rules are messy)
        is_phase_curve = False
        
        # Check explicit star name
        if star_name in self.phase_curve_rules:
            is_phase_curve = True
        # Check (Star, Visit) tuple
        elif (star_name, visit) in self.phase_curve_rules:
            is_phase_curve = True
        
        # 4. DETERMINE FINAL MASK
        if existing_mask:
            # If there is a manual rule, it overrides everything (even phase curve logic)
            # Combine default (or phase default) with existing masks
            
           # Decide what the base mask is
            if is_phase_curve:
                base_start = phase_curve_mask_val
                base_end = phase_curve_mask_val
            else:
                base_start = default_mask_start
                base_end = default_mask_end

            existing_start = existing_mask['mask_start']
            existing_end = existing_mask['mask_end']

            # --- NEW: build (start,end) PAIRS, and only include the base pair if it's real ---
            pairs = []

            # Only include the base/default mask if BOTH ends are set (i.e., not None)
            if base_start is not None and base_end is not None:
                pairs.append((base_start, base_end))

            # Add the manual/existing mask(s)
            if isinstance(existing_start, list):
                if not isinstance(existing_end, list) or len(existing_start) != len(existing_end):
                    raise ValueError("Masking rule has list mask_start but mask_end is not a same-length list.")
                pairs.extend(list(zip(existing_start, existing_end)))
            else:
                pairs.append((existing_start, existing_end))

            # Unpack back into whatever your downstream expects
            mask_start = [s for s, e in pairs]
            mask_end   = [e for s, e in pairs]

            # If only 1 pair, keep it scalar to preserve your current YAML style
            if len(mask_start) == 1:
                mask_start = mask_start[0]
                mask_end   = mask_end[0]

            return {"mask_start": mask_start, "mask_end": mask_end}

        elif is_phase_curve:
            # It's a Phase Curve and no manual rules exist -> Use duration_end
            print(f"  🌙 Identified Phase Curve: {star_name} V{visit if visit else 'N/A'}")
            return {
                'mask_start': phase_curve_mask_val,
                'mask_end': phase_curve_mask_val
            }
            
        else:
            # It's a standard Transit -> Use 30 min default
            return {
                'mask_start': default_mask_start,
                'mask_end': default_mask_end
            }

    
    def get_wavelength_limits_for_config(self, star_name, grating, nrs, visit):
        key = (star_name, grating, nrs, visit)
        if key in self.wavelength_rules:
            return self.wavelength_rules[key]
        key_no_visit = (star_name, grating, nrs, None)
        if key_no_visit in self.wavelength_rules:
            return self.wavelength_rules[key_no_visit]
        return None
    
    def get_detrending_for_config(self, star_name, grating, nrs, visit):
        key = (star_name, grating, nrs, visit)
        if key in self.detrending_rules:
            return self.detrending_rules[key]
        key_no_visit = (star_name, grating, nrs, None)
        if key_no_visit in self.detrending_rules:
            return self.detrending_rules[key_no_visit]
        return None
    
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
    
    def get_transit_time(self, star_name, nrs, grating, visit):
        keys_to_try = []
        if grating == 'SOSS':
            if visit:
                keys_to_try.append(f"{star_name}_NIRISS_V{visit}")
            keys_to_try.append(f"{star_name}_NIRISS")
        else:
            if nrs and visit:
                keys_to_try.append(f"{star_name}_NRS{nrs}_{grating}_V{visit}")
            if nrs:
                keys_to_try.append(f"{star_name}_NRS{nrs}_{grating}")
            if visit:
                keys_to_try.append(f"{star_name}_{grating}_V{visit}")
            keys_to_try.append(f"{star_name}_{grating}")
        for key in keys_to_try:
            if key.upper() in self.transit_times:
                return self.transit_times[key.upper()]
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
        
        detrending_type_for_dir = 'COMPARISON'
        if detrending_config and 'detrending_type' in detrending_config:
            detrend = detrending_config['detrending_type']
            if detrend == 'linear_discontinuity':
                detrending_type_for_dir = 'LINEAR_DISCONTINUITY'
            elif detrend == 'explinear':
                detrending_type_for_dir = 'EXPLINEAR'
            elif detrend == 'linear':
                detrending_type_for_dir = 'LINEAR'
            elif detrend == 'gp':
                detrending_type_for_dir = 'GP'
        
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
        if detrending_config and 'detrending_type' in detrending_config:
            detrending_type = detrending_config['detrending_type']
        
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
            if masking['mask_start'] is not None:
                flags['mask_start'] = masking['mask_start']
            if masking['mask_end'] is not None:
                flags['mask_end'] = masking['mask_end']
        
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
                orders_to_run = [1]
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
    output_directory = "./configs"
    archive_csv = "/scratch/midway3/tfairnington/exoarchive_21nov2025.csv"
    reduction_list_csv = "/scratch/midway3/tfairnington/reduction_list.csv"
    transit_times_file = "/scratch/midway3/tfairnington/transit_times.txt"
    
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
