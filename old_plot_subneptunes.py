import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from exotedrf.stage4 import bin_at_resolution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from astropy import constants as const
from scipy.interpolate import interp1d

k_B = 1.380649e-23
G = 6.6743e-11
R_E = 6371000.0
M_E = 5.9722e24
m_H = 1.6735575e-27
R_S = 6.957e8
mmw = 20
R_J_R_E = (const.R_jup / const.R_earth).value 

# --------------------
# Load and prefilter inputs
# --------------------
targets = pd.read_csv("subneptune_target_list_with_nasa_ps.csv")
targets = targets[(targets["snr"] >= 1.0) & (targets["kmag"] >= 7.8)]

df = pd.read_csv("03_trexolists_extended.csv")
for col in ["st_teff", "pl_massj", "st_logg", "StartTime", "ProprietaryPeriod", "pl_radj"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Event"].isin({"Transit", "PhaseC"})]
df = df[df["st_logg"] < 6]

re_jupe = (const.R_jup / const.R_earth).value
df = df[(df["pl_radj"] * re_jupe >= 1.6) & (df["pl_radj"] * re_jupe <= 5.0)]
df = df[(df["ObservingMode"] != "LRS") & (df["ObservingMode"] != "MRS")]

cutoffs = {0: 46204.0, 6: 46023.0}
threshold = df["ProprietaryPeriod"].map(cutoffs).fillna(45839.0)
df = df[df["StartTime"] <= threshold]

host = df["hostname_nn"].astype(str).str.strip().replace({"nan": ""})
letter = (
    df["letter_nn"].fillna("").astype(str).str.strip()
    .str.replace(r"\.0$", "", regex=True).replace({"nan": ""})
)
df["target_id"] = host + letter

keep_target = df.drop_duplicates(subset="target_id").copy()

df.to_csv("observed_subneptune_targets.csv", index=False)

# --------------------
# Helpers
# --------------------
def _planet_key_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    s = (
        s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.lower()
    )
    s = s.str.replace(r"[^a-z0-9\-\s_]+", "", regex=True)
    s = (
        s.str.replace(r"[-_]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    s = s.str.replace(r"\s*([b-z])$", r"\1", regex=True)
    return s

def normalized_density(df_in: pd.DataFrame, mass_col: str, dens_col: str) -> np.ndarray:
    m = df_in[mass_col].to_numpy(dtype=float)
    rho_p = df_in[dens_col].to_numpy(dtype=float)
    good = np.isfinite(m) & np.isfinite(rho_p) & (m > 0) & (rho_p > 0)
    out = np.full_like(rho_p, np.nan, dtype=float)
    out[good] = rho_p[good] / rho_earthlike_at_mass(m[good])
    return out

# --------------------
# Make RED (df) win over BLUE (targets)
# --------------------
red_df = keep_target.copy()

if "pl_name" in targets.columns and not targets["pl_name"].isna().all():
    targets["match_key"] = _planet_key_series(targets["pl_name"])
else:
    targets["match_key"] = _planet_key_series(targets.get("star", "").astype(str))

if "pl_name" in red_df.columns and not red_df["pl_name"].isna().all():
    red_df["match_key"] = _planet_key_series(red_df["pl_name"])
elif "target_id" in red_df.columns:
    red_df["match_key"] = _planet_key_series(red_df["target_id"])
else:
    h = red_df.get("hostname_nn", red_df.get("hostname", "")).astype(str)
    l = red_df.get("letter_nn", red_df.get("letter", "")).astype(str).str.replace(r"\.0$", "", regex=True)
    red_df["match_key"] = _planet_key_series(h.str.strip() + " " + l.str.strip())

overlap = set(red_df["match_key"])
blue_targets = targets[~targets["match_key"].isin(overlap)].copy()

# --------------------
# Density normalization vs. Earth-like curve
# --------------------
earthdf = pd.read_csv("mass-radius-earth-zeng.csv").sort_values("mass")
M_earth_cgs = const.M_earth.cgs.value
R_earth_cgs = const.R_earth.cgs.value
rho_earthlike_curve = (
    earthdf["mass"] * M_earth_cgs / (4 / 3 * np.pi * (earthdf["radius"] * R_earth_cgs) ** 3)
)

rho_earthlike_fn = interp1d(
    earthdf["mass"].to_numpy(),
    rho_earthlike_curve.to_numpy(),
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)

def rho_earthlike_at_mass(mass_array_earth):
    return rho_earthlike_fn(np.asarray(mass_array_earth, dtype=float))

blue_norm = normalized_density(blue_targets, mass_col="pl_bmasse", dens_col="pl_dens")

red_df = red_df.copy()
red_df["pl_masse_from_j"] = (red_df["pl_massj"].to_numpy(dtype=float) * const.M_jup / const.M_earth).value
red_norm = normalized_density(red_df, mass_col="pl_masse_from_j", dens_col="pl_dens_cgs")

# 2) Utility to build the same match key the table uses
def _planet_key_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    s = s.str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.lower()
    s = s.str.replace(r"[^a-z0-9\-\s_]+", "", regex=True)
    s = s.str.replace(r"[-_]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"\s*([b-z])$", r"\1", regex=True)
    return s

def planet_row(red_df: pd.DataFrame, name_or_id: str) -> pd.Series:
    """Return the *single* row for a planet by flexible identifier."""
    if "match_key" not in red_df.columns:
        # build it the same way you did earlier
        if "pl_name" in red_df.columns and not red_df["pl_name"].isna().all():
            red_df = red_df.assign(match_key=_planet_key_series(red_df["pl_name"]))
        elif "target_id" in red_df.columns:
            red_df = red_df.assign(match_key=_planet_key_series(red_df["target_id"]))
        else:
            h = red_df.get("hostname_nn", red_df.get("hostname", "")).astype(str)
            l = red_df.get("letter_nn", red_df.get("letter", "")).astype(str).str.replace(r"\.0$", "", regex=True)
            red_df = red_df.assign(match_key=_planet_key_series(h.str.strip() + " " + l.str.strip()))

    key = _planet_key_series(pd.Series([name_or_id])).iloc[0]
    rows = red_df.loc[red_df["match_key"] == key]

    if rows.empty:
        # fallback: loose contains match
        rows = red_df.loc[red_df["match_key"].str.contains(key, na=False)]
    if rows.empty:
        raise KeyError(f"No planet matched '{name_or_id}'")

    # if multiple, take the first (or handle as you like)
    return rows.iloc[0]

def planet_params(red_df: pd.DataFrame, name_or_id: str) -> dict:
    r = planet_row(red_df, name_or_id)
    return {
        "match_key": r["match_key"],
        "teq": float(r.get("pl_Teq_K", np.nan)),
        "mass": float(r.get("pl_masse_from_j", np.nan)),
        "rho_norm": float(r.get("rho_norm", np.nan)),
        "rad": float(r.get("pl_radj", np.nan))*R_J_R_E,
        'rstar': float(r.get("st_rad", np.nan)),
        "dens_cgs": float(r.get("pl_dens_cgs", np.nan)),
        "spec_category": r.get("spec_category", "other"),
        "published": bool(r.get("published", False)),
        "Kmag": float(r.get("sy_kmag", np.nan))
    }



'''
df1 = pd.read_csv('spectra/TOI421/TOI421_SPECTRUM/TOI421b_NIRSPEC_G395M_nrs1_R100.csv')
wave1 = df1.wavelength.values
depth1 = df1.depth.values
depth_err1 = df1.depth_err.values
wave1, we1, depth1, depth_err1 = bin_at_resolution(wave1, depth1, depth_err1, 40, 'average')
p1 = planet_params(red_df, "TOI-421 b")
teq1 = p1["teq"]
rp1 = p1["rad"]
mp1 = p1["mass"]
rs1 = p1['rstar']
H1 = (k_B * teq1 * (rp1*R_E)**2)/(mmw * m_H * G * (mp1 * M_E))
mask1 = (wave1 > 4.00) & (wave1 < 4.50)

depth_err1_mask = np.nanmean(depth_err1[mask1])
depth_err1_H = depth_err1_mask / ((2 * (rp1 * R_E) * H1) / (rs1 * R_S)**2)
'''

# ~~~~~~~~~~~~~~~~~~~~~ ERROR IN H CALC ######
def grab_planet(infile, pl_name, R=100, archive=False):
    if archive:
        df1 = pd.read_csv(infile)
        wave1 = df1.CENTRALWAVELNG.values
        depth1 = df1.PL_TRANDEP.values/100
        depth_err1 = df1.PL_TRANDEPERR1.values / 100
    else:
        df1 = pd.read_csv(infile)
        wave1 = df1.wavelength.values
        depth1 = df1.depth.values
        depth_err1 = df1.depth_err.values
    if R == 40:
        pass
    else:
        wave1, we1, depth1, depth_err1 = bin_at_resolution(wave1, depth1, depth_err1, 40, 'average')
    p1 = planet_params(red_df, pl_name)
    teq1 = p1["teq"]
    rp1 = p1["rad"]
    mp1 = p1["mass"]
    rs1 = p1['rstar']
    kmag1 = p1['Kmag']
    H1 = (k_B * teq1 * (rp1*R_E)**2)/(mmw * m_H * G * (mp1 * M_E))
    mask1 = (wave1 > 4.00) & (wave1 < 4.50)
    depth_err1_mask = np.nanmean(depth_err1[mask1])
    depth_err1_H = depth_err1_mask / ((2 * (rp1 * R_E) * H1) / (rs1 * R_S)**2)
    return teq1, depth_err1_H, rp1, mp1, rs1, kmag1

toi421b = grab_planet('spectra/TOI421/TOI421_SPECTRUM/TOI421b_NIRSPEC_G395M_nrs1_R100.csv', "TOI-421 b")
hatp11b = grab_planet('spectra/HATP11/HATP11b_SPECTRUM/HATP11b_NIRSPEC_G395H_nrs2_R200.csv', "HAT-P-11 b")
toi178d = grab_planet('spectra/TOI178/NRS1_FREELD_LINEAR/TOI178d_NIRSPEC_G395M_nrs1_R100.csv', "TOI-178 d")
toi1468c = grab_planet('spectra/TOI1468/NRS2_FREELD_LINEAR/TOI1468c_NIRSPEC_G395H_nrs2_R100.csv', "TOI-1468 c")
ltt3780c = grab_planet('spectra/LTT3780/NRS2_FREELD_LINEAR/LTT3780c_NIRSPEC_G395H_nrs2_R50.csv', "LTT 3780 c")
toi1231b = grab_planet('spectra/TOI1231/NRS2_FREELD_LINEAR/TOI1231b_NIRSPEC_G395H_nrs2_R100.csv', "TOI-1231 b")

k218b_p = grab_planet('K2-18_nirspec_paper.csv', "K2-18 b", archive=True)
lhs1140b_p = grab_planet('LHS-1140_nirspec_paper.csv', "LHS 1140 b", archive=True)
toi836b_p = grab_planet('TOI-836_nirspec_paper.csv', "TOI-836 b", archive=True)
toi776b_p = grab_planet('TOI-776_nirspec_paper.csv', "TOI-776 b", archive=True)
toi776c_p = grab_planet('TOI-776c_nirspec_paper.csv', "TOI-776 c", archive=True)
gj1214b_p = grab_planet('GJ-1214b_nirspec_paper.csv', "GJ 1214 b", archive=True)
toi836c_p = grab_planet('TOI-836c_nirspec_paper.csv', "TOI-836 c", archive=True)
gj3470b_p = grab_planet('GJ-3470b_nircam_paper.csv', "Gj 3470 b", archive=True)


# Create DataFrame from your manually plotted planets
manual_planets_data = []

# Unpublished planets (archive=False)
unpublished_planets = [
    ("HAT-P-11 b", hatp11b), 
    ("TOI-178 d", toi178d),
    ("TOI-1468 c", toi1468c),
    ("LTT 3780 c", ltt3780c),
    ("TOI-1231 b", toi1231b),

]

# Published planets (archive=True)  
published_planets = [
    ("TOI-421 b", toi421b),
    ("K2-18 b", k218b_p),
    ("LHS 1140 b", lhs1140b_p),
    ("TOI-836 b", toi836b_p),
    ("TOI-776 b", toi776b_p),
    ("TOI-776 c", toi776c_p),
    ("GJ 1214 b", gj1214b_p),
    ("TOI-836 c", toi836c_p),
    ("GJ 3470 b", gj3470b_p)
]


for name, data in published_planets:
    manual_planets_data.append({
        'planet_name': name,
        'match_key': _planet_key_series(pd.Series([name])).iloc[0],
        'teq': data[0], 
        'depth_err_H': data[1],
        'rp': data[2],
        'mp': data[3], 
        'rstar': data[4],
        'kmag': data[5],
        'published': True,
        'spec_category': 'clear'  # You can adjust this based on your knowledge
    })

# Create the DataFrame
manual_red_df = pd.DataFrame(manual_planets_data)
# ------- derive symbols from NON-dedup df and map to manual_red_df -------
df_for_symbols = df.copy()  # the full (non-dedup) table

# Ensure a match_key on the non-dedup df
if 'pl_name' in df_for_symbols.columns and not df_for_symbols['pl_name'].isna().all():
    df_for_symbols['match_key'] = _planet_key_series(df_for_symbols['pl_name'])
elif 'target_id' in df_for_symbols.columns:
    df_for_symbols['match_key'] = _planet_key_series(df_for_symbols['target_id'])
else:
    h = df_for_symbols.get('hostname_nn', df_for_symbols.get('hostname', '')).astype(str)
    l = (df_for_symbols.get('letter_nn', df_for_symbols.get('letter', '')).astype(str)
         .str.replace(r'\.0$', '', regex=True))
    df_for_symbols['match_key'] = _planet_key_series(h.str.strip() + ' ' + l.str.strip())

# spectrum → clear/flat/other
spec = df_for_symbols['spectrum'].astype(str) if 'spectrum' in df_for_symbols.columns \
       else pd.Series(index=df_for_symbols.index, dtype=str)

is_clear = spec.str.contains(r'\bclear\b', case=False, na=False)
is_flat  = spec.str.contains(r'\bflat\b',  case=False, na=False)

agg = (
    df_for_symbols
    .assign(__clear=is_clear, __flat=is_flat)
    .groupby('match_key', dropna=False)[['__clear','__flat']]
    .any()
    .reset_index()
)

# precedence: clear > flat > other
cat_map = {}
for _, row in agg.iterrows():
    if row['__clear']:
        cat_map[row['match_key']] = 'clear'
    elif row['__flat']:
        cat_map[row['match_key']] = 'flat'
    else:
        cat_map[row['match_key']] = 'other'

# Map onto manual_red_df (keep any prior values if present)
manual_red_df['spec_category'] = manual_red_df.get('spec_category') \
    .where(manual_red_df.get('spec_category').notna(), None)
manual_red_df['spec_category'] = manual_red_df['match_key'].map(cat_map) \
    .fillna(manual_red_df['spec_category']).fillna('other')

# published flag (from full table)
pub_raw = df_for_symbols.get('published', pd.Series(index=df_for_symbols.index, dtype=object))
is_pub = (
    pub_raw.astype(str).str.strip().str.lower().isin({'yes','y','true','published'})
)
pub_agg = (
    df_for_symbols.assign(__pub=is_pub)
    .groupby('match_key', dropna=False)['__pub']
    .any()
)

# combine: True if manual_red_df already had True OR aggregated True
manual_red_df['published'] = manual_red_df['match_key'].map(pub_agg).fillna(False) | manual_red_df['published'].fillna(False)
spec_str = df_for_symbols.get('spectrum', pd.Series(index=df_for_symbols.index, dtype=object)).astype(str).str.strip().str.lower()
is_pub = (
    df_for_symbols.get('published', pd.Series(index=df_for_symbols.index, dtype=object))
    .astype(str).str.strip().str.lower().isin({'yes','y','true','published'})
)

unsure_unpub_keys = (
    df_for_symbols.loc[spec_str.str.contains(r'\bunsure\b', na=False) & (~is_pub), 'match_key']
    .dropna().drop_duplicates()
)

# Make sure the dedup red table has match_key and the fields we need
red_lookup = red_df.copy()
if 'match_key' not in red_lookup.columns:
    if 'pl_name' in red_lookup.columns and not red_lookup['pl_name'].isna().all():
        red_lookup['match_key'] = _planet_key_series(red_lookup['pl_name'])
    elif 'target_id' in red_lookup.columns:
        red_lookup['match_key'] = _planet_key_series(red_lookup['target_id'])
    else:
        h = red_lookup.get('hostname_nn', red_lookup.get('hostname', '')).astype(str)
        l = (red_lookup.get('letter_nn', red_lookup.get('letter', '')).astype(str)
             .str.replace(r'\.0$', '', regex=True))
        red_lookup['match_key'] = _planet_key_series(h.str.strip() + ' ' + l.str.strip())

cols_needed = [
    'match_key','pl_Teq_K','sy_kmag','pl_radj','pl_massj','st_rad','pl_name','target_id'
]
m = (pd.DataFrame({'match_key': unsure_unpub_keys})
       .merge(red_lookup[cols_needed], on='match_key', how='left'))

# Estimate y the same way you do for blue points using K2-18 b as baseline
baseline_err, baseline_kmag = k218b_p[1], k218b_p[5]
m['depth_err_H'] = baseline_err * 10 ** ((baseline_kmag - m['sy_kmag']) / 2.5)

# X is Teq (same convention as your red points)
m['teq'] = pd.to_numeric(m['pl_Teq_K'], errors='coerce')

# Other optional columns (not used for plotting y, but kept for completeness)
m['rp'] = pd.to_numeric(m['pl_radj'], errors='coerce') * R_J_R_E
m['mp'] = (pd.to_numeric(m['pl_massj'], errors='coerce') * const.M_jup / const.M_earth).value
m['rstar'] = pd.to_numeric(m['st_rad'], errors='coerce')
m['kmag'] = pd.to_numeric(m['sy_kmag'], errors='coerce')

# Name to annotate
m['planet_name'] = m['pl_name'].where(m['pl_name'].notna() & (m['pl_name'].astype(str).str.strip() != ''),
                                      m['target_id']).fillna(m['match_key'])

# Tag as red, unpublished, and 'unsure'
m['published'] = False
m['spec_category'] = 'unsure'

# Keep only rows with finite teq & depth_err_H so they actually plot
m = m[np.isfinite(m['teq']) & np.isfinite(m['depth_err_H'])]

# Append into manual_red_df in the same schema
manual_red_df = pd.concat([
    manual_red_df,
    m[['planet_name','match_key','teq','depth_err_H','rp','mp','rstar','kmag','published','spec_category']]
], ignore_index=True)

# -------------------- PLOT (unchanged logic; only markers + legend updated) --------------------
plt.figure(figsize=(12, 8))

# BLUE targets: keep *exact* logic you had (fixed size, Kmag scaling for err_H)
blue_sc = None
for i in range(len(blue_targets)):
    target_row = blue_targets.iloc[i]
    target_err_H = k218b_p[1] * 10**((k218b_p[5] - target_row['kmag'])/2.5)
    sc = plt.scatter(target_row['scale_temp'], target_err_H, c='b', alpha=0.7, s=30)
    if blue_sc is None:
        blue_sc = sc  # capture one handle for legend

# RED masks (categories) on manual_red_df using the mapped fields
x_r = manual_red_df['teq'].to_numpy(dtype=float)
y_r = manual_red_df['depth_err_H'].to_numpy(dtype=float)
valid_r = np.isfinite(x_r) & np.isfinite(y_r)

# ---- manual overrides for spec_category ----
override_flat_names = ['TOI-1468 c', 'LTT 3780 c', 'TOI-178 d']
override_flat_keys = _planet_key_series(pd.Series(override_flat_names))

manual_red_df.loc[
    manual_red_df['match_key'].isin(override_flat_keys),
    'spec_category'
] = 'flat'


spec_cat = manual_red_df['spec_category'].astype(str).str.lower()
pub = manual_red_df['published'].to_numpy(dtype=bool)

mask_clear = (spec_cat == 'clear') & valid_r
mask_flat  = (spec_cat == 'flat')  & valid_r
mask_other = (spec_cat == 'other') & valid_r

# --- RED base marks (red), categories only ---
# clear: star
red_clear_sc = plt.scatter(x_r[mask_clear], y_r[mask_clear], c='r', marker='*', s=150, zorder=4)
# flat: underscore
red_flat_sc  = plt.scatter(x_r[mask_flat],  y_r[mask_flat],  c='r', marker='_', s=200, linewidths=2, zorder=5)
# other: circle
red_other_sc = plt.scatter(x_r[mask_other], y_r[mask_other], c='r', marker='o', s=80,  zorder=4)

# --- Purple (published) outline overlay, category-aware ---
# For stars & circles: hollow purple edge on top
mask_pub_clear = mask_clear & pub
mask_pub_other = mask_other & pub
plt.scatter(x_r[mask_pub_clear], y_r[mask_pub_clear],
            facecolors='none', edgecolors='purple', marker='*', s=220, linewidths=2, zorder=6)
plt.scatter(x_r[mask_pub_other], y_r[mask_pub_other],
            facecolors='none', edgecolors='purple', marker='o', s=110, linewidths=2, zorder=6)

# For underscores: thicker purple underlay first to give an outline
mask_pub_flat = mask_flat & pub
plt.scatter(x_r[mask_pub_flat], y_r[mask_pub_flat],
            c='purple', marker='_', s=500, linewidths=2, zorder=3)  # underlay

ax = plt.gca()

# ---- Legend: series + shapes + "published" outline (extra labeling you asked for) ----
from matplotlib.lines import Line2D

# (use one real series handle for blue)
if blue_sc is not None:
    blue_sc.set_label('Targets (blue)')

mask_unsure = (spec_cat == 'unsure') & valid_r

# Plot unsure as red circles (same style as "other", per your request to "make them red")
red_unsure_sc = plt.scatter(x_r[mask_unsure], y_r[mask_unsure], c='r', marker='o', s=80, zorder=4)

# --- Update legend to include 'unsure' explicitly ---
shape_handles = [
    Line2D([], [], marker='*', color='r', linestyle='None', markersize=8,  label='Clear (red)'),
    Line2D([], [], marker='_', color='r', linestyle='None', markersize=14, label='Flat (red)'),
    Line2D([], [], marker='o', color='r', linestyle='None', markersize=6,  label='Other/unknown (red)'),
]

# published key (purple outline)
published_handle = Line2D([], [], marker='o', linestyle='None',
                          markerfacecolor='none', markeredgecolor='purple',
                          markersize=6, label='Published (purple outline)')

combined_handles = [blue_sc, *shape_handles, published_handle] if blue_sc is not None else [*shape_handles, published_handle]
combined_labels  = [h.get_label() for h in combined_handles]

ax.legend(combined_handles, combined_labels,  frameon=True, title='Legend', scatterpoints=1)

# ---- Labels (unchanged logic, just colors to match series) ----
text_kw = dict(textcoords='offset points', xytext=(3, 3), ha='left', va='bottom',
               fontsize=8, zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Label manual red planets
for _, row in manual_red_df.loc[valid_r].iterrows():
    plt.annotate(row['planet_name'], (row['teq'], row['depth_err_H']), color='darkred', **text_kw)

# Optional: label blue targets (same as your current logic)
for i in range(len(blue_targets)):
    target_row = blue_targets.iloc[i]
    target_err_H = k218b_p[1] * 10**((k218b_p[5] - target_row['kmag'])/2.5)
    plt.annotate(target_row.get('match_key', f'Target_{i}'),
                 (target_row['scale_temp'], target_err_H),
                 color='navy', **text_kw)

# Axes (unchanged)
plt.xlabel('Equilibrium Temperature (K)')
plt.ylabel('Rel. Depth Error (H, MMW=20)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('err_test_styled.png', dpi=300, bbox_inches='tight')
plt.show()


