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


#for i in range(len(blue_targets)):
#    target_i = target[i]
    # match closest properties to above planets
#    closest_match = target.isclosestto(minimize properties in list(teq1 then closest rp1 and mp1 within +-50K, else +-100K) (toi421b, hatp11b... toi836c_p, gj3470b_p))
#    closest_match_dict = grab_planet(XYZ)
#    match = closest_match_dict[5]
#    target = target_i['kmag']
#    teq1 = target_i['scale_temp']
#    depth_err1_H = closest_match_dict[1](corresponding to depth_err1_H) * 10**((match-target)/2.5)
for i in range(len(blue_targets)):
    target_row = blue_targets.iloc[i]  
    target_err_H = k218b_p[1] * 10**((k218b_p[5] - target_row['kmag'])/2.5)
    plt.scatter(target_row['scale_temp'], target_err_H, c='b', alpha=0.7)


plt.scatter(toi421b[0], toi421b[1], c='r', label='1 Visit')
plt.scatter(hatp11b[0], hatp11b[1], c='r', label='1 Visit')
plt.scatter(toi178d[0], toi178d[1], c='r', label='1 Visit')
plt.scatter(toi1468c[0], toi1468c[1], c='r', label='1 Visit')
plt.scatter(ltt3780c[0], ltt3780c[1], c='r', label='1 Visit')
plt.scatter(toi1231b[0], toi1231b[1], c='r', label='1 Visit')

plt.scatter(k218b_p[0], k218b_p[1], c='r', label='1 Visit')
plt.scatter(lhs1140b_p[0], lhs1140b_p[1], c='r', label='1 Visit')
plt.scatter(toi836b_p[0], toi836b_p[1], c='r', label='2 Visits')
plt.scatter(toi776b_p[0], toi776b_p[1], c='r', label='1 Visit (but has 2)')
plt.scatter(toi776c_p[0], toi776c_p[1], c='r', label='2 Visits')
plt.scatter(gj1214b_p[0], gj1214b_p[1], c='r', label='2 Visits')
plt.scatter(toi836c_p[0], toi836c_p[1], c='r', label='1 Visit')
plt.scatter(gj3470b_p[0], gj3470b_p[1], c='r', label='1 Visit')

plt.xlabel('Equilibrium Temperature (K)')
plt.ylabel('Rel. Depth Error (H, MMW=20)')
plt.savefig('err_test.png')
plt.close()
