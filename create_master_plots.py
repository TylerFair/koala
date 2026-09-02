import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

root = "/scratch/midway3/tfairnington"

def parse_dir(dname):
    parts = dname.split('_')
    planet = parts[0]
    # Updated modes list to catch everything
    modes = ['G395H', 'G395M', 'SOSS', 'PRISM', 'G140H', 'G140M']
    mode = next((p for p in parts if any(m in p for m in modes)), "DATA")
    nrs = next((p for p in parts if 'NRS' in p), "")
    inst_tag = f"{mode}_{nrs}".strip('_')
    v_match = re.search(r'V[0-9]+', dname)
    version = v_match.group() if v_match else None
    return planet, inst_tag, version

def get_data_file(dpath, version):
    if not os.path.exists(dpath): return None
    files = os.listdir(dpath)
    
    # PRIORITY SEARCH: G395H uses Rreference, but PRISM often uses Rnative
    search_terms = ['Rreference', 'Rnative', 'R100']
    
    for term in search_terms:
        if version:
            # Look for term + version (e.g., Rnative_V1.csv)
            pattern = rf"{term}_{version}\.csv$"
        else:
            # Look for term + .csv exactly (e.g., Rnative.csv)
            pattern = rf"{term}\.csv$"
            
        for f in files:
            # Check pattern AND exclude metadata files
            if re.search(pattern, f) and not any(x in f for x in ['params', 'wavelengths', 'lightcurves', 'summary', 'poly_coeffs']):
                 return os.path.join(dpath, f)
                 
    return None

def calculate_weighted_offset(anchor_df, target_df):
    min_wave = max(anchor_df['wavelength'].min(), target_df['wavelength'].min())
    max_wave = min(anchor_df['wavelength'].max(), target_df['wavelength'].max())
    
    # PRISM/SOSS overlap is large; G395H overlap is small. 
    # If overlap is tiny (<0.02 micron), assume no overlap (delta = 0)
    if (max_wave - min_wave) < 0.02: return 0.0
    
    a_ov = anchor_df[(anchor_df['wavelength'] >= min_wave) & (anchor_df['wavelength'] <= max_wave)]
    t_ov = target_df[(target_df['wavelength'] >= min_wave) & (target_df['wavelength'] <= max_wave)]
    
    if len(a_ov) < 2 or len(t_ov) < 2: return 0.0

    a_weights = 1.0 / (a_ov['depth_err00']**2)
    t_weights = 1.0 / (t_ov['depth_err00']**2)
    a_mean = np.sum(a_ov['depth00'] * a_weights) / np.sum(a_weights)
    t_mean = np.sum(t_ov['depth00'] * t_weights) / np.sum(t_weights)
    
    return a_mean - t_mean

# 1. Map directories
all_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and "_" in d]
planet_map = {}
for d in all_dirs:
    p, inst, v = parse_dir(d)
    if p not in planet_map: planet_map[p] = []
    planet_map[p].append({'name': d, 'version': v, 'inst': inst})

# 2. Process each planet
for planet, folder_list in planet_map.items():
    versions_present = list(set(info['version'] for info in folder_list))
    versions_to_run = [v for v in versions_present if v is not None] or [None]

    for current_v in versions_to_run:
        all_data_info = []

        for info in folder_list:
            if info['version'] == current_v or info['version'] is None:
                csv_path = get_data_file(os.path.join(root, info['name']), info['version'])
                if csv_path:
                    try:
                        df = pd.read_csv(csv_path).sort_values('wavelength')
                        all_data_info.append({'df': df, 'inst': info['inst'], 'path': os.path.join(root, info['name'])})
                    except: continue

        if not all_data_info: 
            # Optional: Uncomment to debug specifically which folders are empty
            # print(f"Skipping {planet} {current_v}: No CSV found.")
            continue

        # --- ALIGNMENT LOGIC ---
        aligned_data = []
        instruments_found = []
        target_folders = []

        if len(all_data_info) > 1:
            # Multi-instrument: Stitch and Align
            # Anchor priority: SOSS > PRISM > First available
            anchor_idx = 0
            for priority in ['SOSS', 'PRISM']:
                for i, info in enumerate(all_data_info):
                    if priority in info['inst']:
                        anchor_idx = i
                        break
                else: continue
                break
            
            anchor_df = all_data_info[anchor_idx]['df']
            
            for info in all_data_info:
                offset = 0.0
                if info['inst'] != all_data_info[anchor_idx]['inst']:
                    offset = calculate_weighted_offset(anchor_df, info['df'])
                
                shifted_df = info['df'].copy()
                shifted_df['depth00'] = info['df']['depth00'] + offset
                aligned_data.append(shifted_df)
                instruments_found.append(info['inst'])
                target_folders.append(info['path'])
        else:
            # Single instrument (e.g. WASP-6 PRISM): Just plot
            aligned_data.append(all_data_info[0]['df'])
            instruments_found.append(all_data_info[0]['inst'])
            target_folders.append(all_data_info[0]['path'])

        # 3. Dynamic Tagging
        tag_list = sorted(list(set(instruments_found)))
        v_str = f"_{current_v}" if current_v else ""
        name_tag = f"{planet}_{'_'.join(tag_list)}{v_str}"

        # 4. Plotting
        plt.figure(figsize=(10, 5))
        for data in aligned_data:
            plt.errorbar(data['wavelength'], data['depth00'], yerr=data['depth_err00'], 
                         fmt='o', ls='', ms=4, mfc='white', mec='k', ecolor='k', mew=1.2, capsize=0)

        plt.xlabel(r"Wavelength ($\mu$m)", fontsize=13)
        plt.ylabel(r"Transit Depth $(R_p/R_s)^2$", fontsize=13)
        plt.gca().tick_params(direction='in', top=True, right=True)

        # 5. Save
        idx = "42" if current_v == "V2" else "41"
        final_filename = f"{idx}_spectrum_{name_tag}.png"
        
        for save_folder in target_folders:
            plt.savefig(os.path.join(save_folder, final_filename), bbox_inches='tight', dpi=300)
        
        plt.close()
        print(f"Generated: {final_filename}")
