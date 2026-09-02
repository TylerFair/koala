# --- Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from astropy import constants as const
from scipy.interpolate import interp1d
from exotedrf.stage4 import bin_at_resolution

# --- Constants
k_B   = 1.380649e-23
G     = 6.6743e-11
R_E   = 6_371_000.0
M_E   = 5.9722e24
m_H   = 1.6735575e-27
R_S   = 6.957e8
R_J_R_E = (const.R_jup / const.R_earth).value
mmw   = 18.0

### NEW/CHANGED: plot color constants (human-editable)
COLOR_RED     = "r"
COLOR_BLUE    = "b"
COLOR_PURPLE  = "purple"
COLOR_ORANGE  = "orange"   # “orangey orange” for unpublished non-NIRISS & all NIRSpec points

# --------------------
# Helpers
# --------------------
def planet_key(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = (s.str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.lower()
           .str.replace(r"[^a-z0-9\-\s_]+", "", regex=True)
           .str.replace(r"[-_]+", " ", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.replace(r"\s*([b-z])$", r"\1", regex=True))
    return s

def ensure_match_key(df: pd.DataFrame) -> pd.DataFrame:
    if "match_key" in df.columns:
        return df
    if "pl_name" in df.columns and not df["pl_name"].isna().all():
        return df.assign(match_key=planet_key(df["pl_name"]))
    if "target_id" in df.columns:
        return df.assign(match_key=planet_key(df["target_id"]))
    h = df.get("hostname_nn", df.get("hostname", "")).astype(str)
    l = df.get("letter_nn",   df.get("letter",   "")).astype(str).str.replace(r"\.0$", "", regex=True)
    return df.assign(match_key=planet_key(h.str.strip() + " " + l.str.strip()))

def upper_except_last_letter(name: str) -> str:
    s = str(name or "")
    if not s: return s
    last = max((i for i,c in enumerate(s) if c.isalpha()), default=None)
    if last is None: return s
    out = [ (c.lower() if i == last else (c.upper() if c.isalpha() else c)) for i,c in enumerate(s) ]
    return "".join(out)

def rho_earthlike_fn_from_file(csv_path: str):
    df = pd.read_csv(csv_path).sort_values("mass")
    M_earth_cgs = const.M_earth.cgs.value
    R_earth_cgs = const.R_earth.cgs.value
    rho = df["mass"] * M_earth_cgs / (4/3*np.pi * (df["radius"]*R_earth_cgs)**3)
    return interp1d(df["mass"].to_numpy(), rho.to_numpy(), kind="linear",
                    bounds_error=False, fill_value=np.nan, assume_sorted=True)

def normalized_density(df_in: pd.DataFrame, mass_col: str, dens_col: str, rho_fn) -> np.ndarray:
    m = df_in[mass_col].to_numpy(float)
    rho_p = df_in[dens_col].to_numpy(float)
    good = np.isfinite(m) & np.isfinite(rho_p) & (m > 0) & (rho_p > 0)
    out = np.full_like(rho_p, np.nan, dtype=float)
    out[good] = rho_p[good] / rho_fn(m[good])
    return out

def planet_row(red_df: pd.DataFrame, name_or_id: str) -> pd.Series:
    red_df = ensure_match_key(red_df)
    key = planet_key(pd.Series([name_or_id])).iloc[0]
    rows = red_df.loc[red_df["match_key"] == key]
    if rows.empty:
        rows = red_df.loc[red_df["match_key"].str.contains(key, na=False)]
    if rows.empty:
        raise KeyError(f"No planet matched '{name_or_id}'")
    return rows.iloc[0]

def planet_params(red_df: pd.DataFrame, name_or_id: str) -> dict:
    r = planet_row(red_df, name_or_id)
    return {
        "match_key": r["match_key"],
        "teq": float(r.get("pl_Teq_K", np.nan)),
        "mass": float(r.get("pl_masse_from_j", np.nan)),
        "rho_norm": float(r.get("rho_norm", np.nan)),
        "rad": float(r.get("pl_radj", np.nan)) * R_J_R_E,
        "rstar": float(r.get("st_rad", np.nan)),
        "dens_cgs": float(r.get("pl_dens_cgs", np.nan)),
        "spec_category": r.get("spec_category", "other"),
        "published": bool(r.get("published", False)),
        "Kmag": float(r.get("sy_kmag", np.nan)),
        "Jmag": float(r.get("sy_jmag", np.nan)),
        "pl_trandur": float(r.get("pl_trandur", np.nan)),
    }

def scale_height(teq, rp_re, mp_me):
    return (k_B * teq * (rp_re*R_E)**2) / (mmw * m_H * G * (mp_me * M_E))

def depth_err_H_from_err(err, teq, rp_re, mp_me, rstar_rs):
    H = scale_height(teq, rp_re, mp_me)
    return err / ((2 * (rp_re * R_E) * 5 * H) / (rstar_rs * R_S)**2)

def scale_height_normalized_spectrum(depths, waves, teq, rp_re, mp_me, rstar_rs):
    H = scale_height(teq, rp_re, mp_me)
    mask_H2O = (waves >= 1.36) & (waves <=1.44)
    mask_J = (waves >= 1.24) & (waves <=1.30)
    med_depths = np.nanmean(depths[mask_H2O]) - np.nanmean(depths[mask_J])
    A_H = med_depths / ((2 * (rp_re * R_E) * 5 * H) / (rstar_rs * R_S)**2)
    return depths / A_H

def grab_planet(infile, pl_name, n_visits=1, R=100, archive=False):
    df1 = pd.read_csv(infile)
    if archive:
        wave, depth, derr = df1.CENTRALWAVELNG.values, df1.PL_TRANDEP.values/100, df1.PL_TRANDEPERR1.values/100
    else:
        wave, depth, derr = df1.wavelength.values, df1.depth.values, df1.depth_err.values
    if R != 40: 
        wave, _, depth, derr = bin_at_resolution(wave, depth, derr, 40, "average")

    p = planet_params(red_df, pl_name)
    H = scale_height(p["teq"], p["rad"], p["mass"])
    mask = (wave > 1.3) & (wave < 1.5)
    err_band = np.nanmean(derr[mask])  * (1/np.sqrt(n_visits))
    
    N_H = (err_band * (p["rstar"] * R_S)**2) / (2 * (p["rad"] * R_E) * 5 * H)

    depth_err_H = depth_err_H_from_err(err_band, p["teq"], p["rad"], p["mass"], p["rstar"])
    #if pl_name == "K2-18 b":  
    #    depth_err_H = depth_err_H
    return p["teq"], depth_err_H, p["rad"], p["mass"], p["rstar"], p["Jmag"], err_band, N_H

def manual_planet(error_ppm, pl_name, n_visits=1):
    p  = planet_params(red_df, pl_name)
    err = error_ppm*1e-6 * (1/np.sqrt(n_visits))
    depth_err_H = depth_err_H_from_err(err, p["teq"], p["rad"], p["mass"], p["rstar"])
    H = scale_height(p["teq"], p["rad"], p["mass"])
    N_H = (err * (p["rstar"] * R_S)**2) / (2 * (p["rad"] * R_E) * H)
    return p["teq"], depth_err_H, p["rad"], p["mass"], p["rstar"], p["Jmag"], err, N_H

# --------------------
# Load + prefilter
# --------------------
targets = pd.read_csv("subneptune_target_list_with_nasa_ps_filled.csv")
targets = targets[(targets["snr"] >= 1.0) & (targets["sy_jmag"] >= 7.8) & (targets["st_teff"] <= 4600)]
df = pd.read_csv("03_trexolists_extended_with_nasa_ps_filled.csv")

for col in ["st_teff", "pl_massj", "st_logg", "StartTime", "ProprietaryPeriod", "pl_radj"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Event"].isin({"Transit", "PhaseC"}) & (df["st_logg"] < 6)]
df = df[(df["pl_radj"] * R_J_R_E >= 1.6) & (df["pl_radj"] * R_J_R_E <= 5.0)]
df = df[~df["ObservingMode"].isin(["LRS", "MRS"])]

cutoffs  = {0: 46204.0, 6: 46023.0}
threshold = df["ProprietaryPeriod"].map(cutoffs).fillna(45839.0)
df = df[df["StartTime"] <= threshold]

host  = df["hostname_nn"].astype(str).str.strip().replace({"nan": ""})
letter= (df["letter_nn"].fillna("").astype(str).str.strip()
           .str.replace(r"\.0$", "", regex=True).replace({"nan": ""}))
df["target_id"] = host + letter

keep_target = df.drop_duplicates(subset="target_id").copy()
df.to_csv("observed_subneptune_targets.csv", index=False)

# --------------------
# Densities + match keys
# --------------------
earth_rho_fn = rho_earthlike_fn_from_file("mass-radius-earth-zeng.csv")

targets = ensure_match_key(
    targets.assign(match_key=(planet_key(targets["pl_name"]) if "pl_name" in targets.columns and not targets["pl_name"].isna().all()
                              else planet_key(targets.get("star","").astype(str))))
)

red_df = ensure_match_key(keep_target.copy())

blue_targets = targets[~targets["match_key"].isin(set(red_df["match_key"]))].copy()

# normalized densities
blue_targets["rho_norm"] = normalized_density(blue_targets, "pl_bmasse", "pl_dens", earth_rho_fn)
red_df["pl_masse_from_j"] = (red_df["pl_massj"].to_numpy(float) * const.M_jup / const.M_earth).value
red_df["rho_norm"] = normalized_density(red_df, "pl_masse_from_j", "pl_dens_cgs", earth_rho_fn)

# --------------------
# Build manual/published planet set
# --------------------

#toi421b   = grab_planet('spectra/TOI-421/SPECTRUM/TOI421b_NIRISS_SOSS_order1_R100.csv', "TOI-421 b")
toi421b   = manual_planet(25, "TOI-421 b", n_visits=1) # 1 visit

hatp11b   = grab_planet('spectra/HAT-P-11/SPECTRUM/HATP11b_NIRISS_SOSS_order1_R100.csv', "HAT-P-11 b") # 1 visit
toi1130b  = grab_planet('spectra/TOI-1130/ORDER1_FIXEDLD_LINEAR/TOI1130b_NIRISS_SOSS_order1_R100.csv', "TOI-1130 b") # 1 visit
k218b_p   = grab_planet('K2-18b_niriss_paper.csv', "K2-18 b", archive=True) # 1 visit

gj9827d   = grab_planet('GJ-9827d_niriss_paper.csv', "GJ 9827 d",  archive=True) # joint 2 visits on exoarchive
lhs1140b  = grab_planet('LHS-1140b_niriss_paper.csv', "LHS-1140 b", archive=True, n_visits=1) # 3 visits, 2 more planned in 2026 post cycle5?

toi270d   = manual_planet(25, "TOI-270 d", n_visits=1) # 2 visits
gj3090b   = manual_planet(50, "GJ-3090 b", n_visits=1) # 2 visits

# --------------------
# Reference values for scaling (from K2-18 b spectrum)
# --------------------
k218_ref_err  = float(k218b_p[6])
k218_ref_jmag = float(k218b_p[5])
k218_ref_NH   = float(k218b_p[7])

# --------------------
# NIRSpec (names only; compute y like unsure_unpub)
# --------------------
nirspec_names = [
    "TOI-178 d",   # flat, unpublished
    "TOI-1468 c",  # flat, unpublished
    "LTT 3780 c",  # flat, unpublished
    "TOI-1231 b",  # clear, unpublished
    "LP 791-18 c", # clear, unpublished
    "TOI-836 b",   # flat,  published
    "TOI-776 b",   # flat,  published
    "TOI-776 c",   # flat,  published
    "GJ 1214 b",   # flat,  published
    "TOI-836 c",   # flat,  published
    "GJ 3470 b",   # clear, published
]

# Spectrum class & pub flags for NIRSpec targets (human-editable)
nirspec_spec_map = {
    "TOI-178 d": "flat",
    "TOI-1468 c": "flat",
    "LTT 3780 c": "flat",
    "TOI-1231 b": "clear",
    "LP 791-18 c": "clear",
    "TOI-836 b": "flat",
    "TOI-776 b": "flat",
    "TOI-776 c": "flat",
    "GJ 1214 b": "flat",
    "TOI-836 c": "flat",
    "GJ 3470 b": "clear",
}
nirspec_pub_map = {
    "TOI-178 d": False,
    "TOI-1468 c": False,
    "LTT 3780 c": False,
    "TOI-1231 b": False,
    "LP 791-18 c": False,
    "TOI-836 b": True,
    "TOI-776 b": True,
    "TOI-776 c": True,
    "GJ 1214 b": True,
    "TOI-836 c": True,
    "GJ 3470 b": True,
}

def _friendly_numeric(v):
    return pd.to_numeric(v, errors="coerce")

def nirspec_row_from_red(name: str):
    r = planet_row(red_df, name)
    teq   = _friendly_numeric(r.get("pl_Teq_K"))
    jmag  = _friendly_numeric(r.get("sy_jmag"))
    rp    = _friendly_numeric(r.get("pl_radj")) * R_J_R_E
    mp    = _friendly_numeric(r.get("pl_massj")) * (const.M_jup/const.M_earth).value
    rstar = _friendly_numeric(r.get("st_rad"))
    dur   = _friendly_numeric(r.get("pl_trandur"))

    H = scale_height(teq, rp, mp)
    err_band = (k218_ref_err * 10.0 ** ((jmag - k218_ref_jmag)/5)) * (1/np.sqrt(dur))
    depth_err_H = err_band / ((2 * (rp * R_E) * 5 * H) / (rstar * R_S)**2)

    return {
        "planet_name": name,
        "match_key": planet_key(pd.Series([name])).iloc[0],
        "teq": teq,
        "depth_err_H": depth_err_H,
        "rp": rp,
        "mp": mp,
        "rstar": rstar,
        "jmag": jmag,
        "published": bool(nirspec_pub_map.get(name, False)),
        "spec_category": nirspec_spec_map.get(name, "other"),
        "is_nirspec": True,
    }

nirspec_df = pd.DataFrame([nirspec_row_from_red(n) for n in nirspec_names])

published_planets = [
    ("K2-18 b", k218b_p),
    ("TOI-270 d", toi270d),
    ("GJ 9827 d", gj9827d),
    ("LHS-1140 b", lhs1140b),
    ("TOI-421 b", toi421b),
    ("GJ-3090 b", gj3090b),
]
unpublished_planets = [
    ("HAT-P-11 b", hatp11b),
    ("TOI-1130 b", toi1130b),
]

manual_planets_data = []

def _add_manual(name, data, published_flag):
    manual_planets_data.append({
        "planet_name": name,
        "match_key": planet_key(pd.Series([name])).iloc[0],
        "teq": data[0], "depth_err_H": data[1], "rp": data[2], "mp": data[3],
        "rstar": data[4], "jmag": data[5],
        "published": bool(published_flag),
        "spec_category": "other",   # will be overwritten by cat_map below
        "is_nirspec": False,
    })

for name, data in published_planets:
    _add_manual(name, data, True)

for name, data in unpublished_planets:
    _add_manual(name, data, False)

manual_red_df = pd.DataFrame(manual_planets_data)


# Spectrum categories from full df (clear/flat/other) and published flags
df_for_symbols = ensure_match_key(df.copy())
spec = df_for_symbols["spectrum"].astype(str) if "spectrum" in df_for_symbols.columns else pd.Series(index=df_for_symbols.index, dtype=str)
is_clear = spec.str.contains(r"\bclear\b", case=False, na=False)
is_flat  = spec.str.contains(r"\bflat\b",  case=False, na=False)
agg = (df_for_symbols.assign(__clear=is_clear, __flat=is_flat)
       .groupby("match_key", dropna=False)[["__clear","__flat"]].any().reset_index())

cat_map = {mk: ("clear" if c else "flat" if f else "other") for mk, c, f in agg[["match_key","__clear","__flat"]].itertuples(index=False)}
manual_red_df["spec_category"] = manual_red_df["match_key"].map(cat_map).fillna("other")

pub_agg = (df_for_symbols.assign(__pub=df_for_symbols.get("published","").astype(str)
                                 .str.strip().str.lower().isin({"yes","y","true","published"}))
           .groupby("match_key", dropna=False)["__pub"].any())
manual_red_df["published"] = manual_red_df["match_key"].map(pub_agg).fillna(False) | manual_red_df["published"].fillna(False)

# --------------------
# Unsure + unpublished candidates to extend red set
# --------------------
is_unsure = spec.str.contains(r"\bunsure\b", case=False, na=False)
unsure_agg = df_for_symbols.assign(__unsure=is_unsure).groupby("match_key", dropna=False)["__unsure"].any()
red_df = ensure_match_key(red_df).copy()
red_df["published"]     = red_df["match_key"].map(pub_agg).fillna(False)
red_df["spec_category"] = red_df["match_key"].map(cat_map).fillna("other")
red_df["is_unsure"]     = red_df["match_key"].map(unsure_agg).fillna(False)

unsure_unpub = red_df[(red_df["is_unsure"]) & (~red_df["published"])].copy()
already = set(manual_red_df["match_key"])
unsure_unpub = unsure_unpub[~unsure_unpub["match_key"].isin(already)]

def friendly_name(row):
    if pd.notna(row.get("pl_name")) and str(row["pl_name"]).strip(): return str(row["pl_name"])
    if "target_id" in row and pd.notna(row["target_id"]) and str(row["target_id"]).strip(): return str(row["target_id"])
    h = str(row.get("hostname_nn", row.get("hostname", ""))).strip()
    l = str(row.get("letter_nn",   row.get("letter",   ""))).strip().replace(".0","")
    return (h + " " + l).strip()

for col in ["pl_Teq_K","sy_jmag","pl_trandur","pl_radj","pl_massj","st_rad"]:
    unsure_unpub[col] = pd.to_numeric(unsure_unpub.get(col), errors="coerce")

unsure_unpub = unsure_unpub.assign(
    planet_name = unsure_unpub.apply(friendly_name, axis=1),
    teq  = unsure_unpub["pl_Teq_K"],
    jmag = unsure_unpub["sy_jmag"],
    rp   = unsure_unpub["pl_radj"] * R_J_R_E,
    mp   = unsure_unpub["pl_massj"] * (const.M_jup/const.M_earth).value,
    rstar= unsure_unpub["st_rad"],
    pl_trandur = unsure_unpub["pl_trandur"],
    published=False,
    spec_category="other",
    is_nirspec=False,
)

H_u = scale_height(unsure_unpub["teq"], unsure_unpub["rp"], unsure_unpub["mp"])
err_band_u = (k218_ref_err * 10.0 ** ((unsure_unpub["jmag"] - k218_ref_jmag)/5)) * (1/np.sqrt(unsure_unpub["pl_trandur"]))
unsure_unpub["depth_err_H"] = ( err_band_u /
    ((2 * (unsure_unpub["rp"] * R_E) * 5 *  H_u) / (unsure_unpub["rstar"] * R_S)**2) )
cols = ["planet_name","match_key","teq","depth_err_H","rp","mp","rstar","jmag","published","spec_category","is_nirspec"]

# --------------------
# Combine sets
# --------------------
extended_red_df = pd.concat([manual_red_df, unsure_unpub[cols], nirspec_df], ignore_index=True, sort=False)

# Preserve your manual override for flats
override_flat = planet_key(pd.Series(['TOI-1468 c', 'LTT 3780 c', 'TOI-178 d'])).tolist()
extended_red_df.loc[extended_red_df["match_key"].isin(override_flat), "spec_category"] = "flat"

# --------------------
# PLOTTING
# --------------------
plt.figure(figsize=(12, 8))

blue_sc = None
for r in blue_targets.itertuples(index=False):
    err = k218_ref_err * 10 ** ((getattr(r, "sy_jmag") - k218_ref_jmag)/5) * (1/np.sqrt(getattr(r, "pl_trandur")))
    H   = scale_height(getattr(r, "scale_temp"), getattr(r, "pl_rade"), getattr(r, "pl_bmasse"))
    depth_err_H = ( err / ((2 * (getattr(r, "pl_rade") * R_E) * 5 * H) / (getattr(r, "st_rad") * R_S)**2) )

    if blue_sc is None:
        blue_sc = plt.scatter(getattr(r, "scale_temp"), 1/depth_err_H, c=COLOR_BLUE, alpha=0.7, s=30, label="Potential Targets")
    else:
        plt.scatter(getattr(r, "scale_temp"), 1/depth_err_H, c=COLOR_BLUE, alpha=0.7, s=30)

    plt.annotate(upper_except_last_letter(getattr(r, "match_key", "Target")),
                 (getattr(r, "scale_temp"), 1/depth_err_H),
                 color="navy", textcoords="offset points", xytext=(3,3),
                 ha="left", va="bottom", fontsize=8,
                 zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# RED/ORANGE points by category
x = extended_red_df["teq"].to_numpy(float)
y = extended_red_df["depth_err_H"].to_numpy(float)
valid = np.isfinite(x) & np.isfinite(y)
spec = extended_red_df["spec_category"].astype(str).str.lower()
pub  = extended_red_df["published"].to_numpy(bool)
is_nirspec = extended_red_df["is_nirspec"].fillna(False).to_numpy(bool)

# Masks by spec
mask_clear = (spec == "clear") & valid
mask_flat  = (spec == "flat")  & valid
mask_other = (spec == "other") & valid

# Color rule:
# - ORANGE for:
#    (a) any NIRSpec item (regardless of pub), using star/dash according to spec_category
#    (b) non-published "other" points (previously red circles)
# - RED for the rest
is_orange = (is_nirspec) #| ((~pub) & mask_other)
is_red    = ~is_orange

# Plot CLEAR
plt.scatter(x[mask_clear & is_red],    1/y[mask_clear & is_red],    c=COLOR_RED,    marker="*", s=150, zorder=4)
plt.scatter(x[mask_clear & is_orange], 1/y[mask_clear & is_orange], c=COLOR_ORANGE, marker="*", s=150, zorder=4)

# Plot FLAT
plt.scatter(x[mask_flat & is_red],     1/y[mask_flat & is_red],     c=COLOR_RED,    marker="_", s=200, linewidths=2, zorder=5)
plt.scatter(x[mask_flat & is_orange],  1/y[mask_flat & is_orange],  c=COLOR_ORANGE, marker="_", s=200, linewidths=2, zorder=5)

# Plot OTHER (circles)
plt.scatter(x[mask_other & is_red],    1/y[mask_other & is_red],    c=COLOR_RED,    marker="o", s=80,  zorder=4)
plt.scatter(x[mask_other & is_orange], 1/y[mask_other & is_orange], c=COLOR_ORANGE, marker="o", s=80,  zorder=4)

# Published overlays (same look, applied whether the base is red or orange)
plt.scatter(x[mask_clear & pub], 1/y[mask_clear & pub], facecolors="none", edgecolors=COLOR_PURPLE,
            marker="*", s=220, linewidths=2, zorder=6)
plt.scatter(x[mask_other & pub], 1/y[mask_other & pub], facecolors="none", edgecolors=COLOR_PURPLE,
            marker="o", s=110, linewidths=2, zorder=6)
plt.scatter(x[mask_flat & pub],  1/y[mask_flat & pub],  c=COLOR_PURPLE, marker="_", s=500, linewidths=2, zorder=3)

# annotate red/orange points
text_kw = dict(textcoords="offset points", xytext=(3,3), ha="left", va="bottom",
               fontsize=8, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# orange  (NIRSpec or non-published "other") — bottom
for _, row in extended_red_df.loc[valid & is_orange].iterrows():
    plt.annotate(
        upper_except_last_letter(row["planet_name"]),
        (row["teq"], 1/row["depth_err_H"]),
        color="darkorange",
        zorder=6,  # bottom
        **text_kw
    )

# red  — top
for _, row in extended_red_df.loc[valid & is_red].iterrows():
    plt.annotate(
        upper_except_last_letter(row["planet_name"]),
        (row["teq"], 1/row["depth_err_H"]),
        color="darkred",
        zorder=8,  # top
        **text_kw
    )
# Legend (add orange cohort explanation)
ax = plt.gca()
handles = [
    *( [blue_sc] if blue_sc is not None else [] ),
    Line2D([], [], marker="*", color=COLOR_RED,    linestyle="None", markersize=8,  label="Approved - Features"),
    Line2D([], [], marker="_", color=COLOR_RED,    linestyle="None", markersize=14, label="Approved - Flat"),
    Line2D([], [], marker="o", color=COLOR_RED,    linestyle="None", markersize=6,  label="Approved - Unsure/Unknown"),
  #  Line2D([], [], marker="*", color=COLOR_ORANGE, linestyle="None", markersize=8,  label="NIRSpec/Orange (clear)"),
   # Line2D([], [], marker="_", color=COLOR_ORANGE, linestyle="None", markersize=14, label="NIRSpec/Orange (flat)"),
    Line2D([], [], marker="o", color=COLOR_ORANGE, linestyle="None", markersize=6,  label="No NIRISS - NIRSpec/NIRCam Only"),
    Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor=COLOR_PURPLE, markersize=6, label="Published NIRISS (purple outline)"),
]
labels = [h.get_label() for h in handles if h is not None]
ax.legend([h for h in handles if h is not None], labels, frameon=True, title="Legend", scatterpoints=1)

plt.xlabel("Equilibrium Temperature (K)")
plt.ylabel("Signal/Noise of a 5 Scale Height H2O @ mu=18 amu (NIRISS 1.3-1.5µm)")
plt.yscale("log")
plt.tight_layout()
plt.savefig("niriss_neptune_scales.png", dpi=300, bbox_inches="tight")
plt.show()


rows = []

# Collect BLUE (Potential Targets) points used in the plot
# Recompute here to capture exactly what's plotted (same as in the loop)
for r in blue_targets.itertuples(index=False):
    teq = getattr(r, "scale_temp")
    err = k218_ref_err * 10.0 ** ((getattr(r, "sy_jmag") - k218_ref_jmag)/5) * (1/np.sqrt(getattr(r, "pl_trandur")))
    H   = scale_height(getattr(r, "scale_temp"), getattr(r, "pl_rade"), getattr(r, "pl_bmasse"))
    depth_err_H = err / ((2 * (getattr(r, "pl_rade") * R_E) * 5* H) / (getattr(r, "st_rad") * R_S)**2)
    y_plot = 1.0 / depth_err_H  # this is what's on the plot
    rows.append({
        "planet_name": upper_except_last_letter(getattr(r, "match_key", "Target")),
        "category": "Potential Targets (blue)",
        "published": False,
        "spec_category": "other",
        "is_nirspec": False,
        "x_eq_temp_K": float(teq),
        "y_snr_per_1H_per_visit": float(y_plot),
    })

# Helper to map your red/orange legend categories
def _category_label(spec_cat, is_ns):
    if bool(is_ns):
        return "No NIRISS - NIRSpec/NIRCam Only (orange)"
    spec_cat = (spec_cat or "").lower()
    if spec_cat == "clear":
        return "Approved - Features (red)"
    if spec_cat == "flat":
        return "Approved - Flat (red)"
    return "Approved - Unsure/Unknown (red)"

# Collect RED/ORANGE points from extended_red_df
for r in extended_red_df.itertuples(index=False):
    if not np.isfinite(r.teq) or not np.isfinite(r.depth_err_H):
        continue
    y_plot = 1.0 / float(r.depth_err_H)
    rows.append({
        "planet_name": upper_except_last_letter(getattr(r, "planet_name", "")),
        "category": _category_label(getattr(r, "spec_category", ""), getattr(r, "is_nirspec", False)),
        "published": bool(getattr(r, "published", False)),
        "spec_category": getattr(r, "spec_category", ""),
        "is_nirspec": bool(getattr(r, "is_nirspec", False)),
        "x_eq_temp_K": float(r.teq),
        "snr_per_5H_per_visit": float(y_plot),
    })

# Build DataFrame, add requested calc: (3/(y * 5))^2
table_df = pd.DataFrame(rows)


# Round numeric columns for readability in the CSV/printout (keeps full precision in memory if you want)
rounded = table_df.copy()
rounded["x_eq_temp_K"] = rounded["x_eq_temp_K"].round(1)
rounded["snr_per_5H_per_visit"] = rounded["snr_per_5H_per_visit"].round(3)

# Save and preview
rounded.to_csv("niriss_neptune_scales_table.csv", index=False)
print("\nSaved table to niriss_neptune_scales_table.csv\n")
print(rounded.head(20).to_string(index=False))


def _status_label(is_blue: bool, is_nirspec: bool) -> str:
    if is_blue:
        return "Potential target (blue)"
    return "JWST observed (NIRSpec/NIRCam only) — orange" if is_nirspec else "JWST observed (NIRISS) — red"

# Build lookup dicts from red_df for missing values in extended_red_df
_rn = lambda s: pd.to_numeric(red_df.get(s), errors="coerce")
mk = ensure_match_key(red_df)["match_key"].astype(str)

map_sy_jmag  = pd.Series(_rn("sy_jmag").values,   index=mk).to_dict()
map_st_rad   = pd.Series(_rn("st_rad").values,    index=mk).to_dict()     # Rsun
map_pl_radj  = pd.Series(_rn("pl_radj").values,   index=mk).to_dict()     # Rjup
map_pl_massj = pd.Series(_rn("pl_massj").values,  index=mk).to_dict()     # Mjup
map_trandur  = pd.Series(_rn("pl_trandur").values,index=mk).to_dict()     # hours

# unit converters for fallbacks
RJ_TO_RE = R_J_R_E
MJ_TO_ME = (const.M_jup / const.M_earth).value

rows = []

# 1) BLUE (Potential Targets) — you already have all inputs here
for r in blue_targets.itertuples(index=False):
    teq       = float(getattr(r, "scale_temp"))
    jmag      = float(getattr(r, "sy_jmag"))
    dur       = float(getattr(r, "pl_trandur"))
    rp_re     = float(getattr(r, "pl_rade"))     # Earth radii
    mp_me     = float(getattr(r, "pl_bmasse"))   # Earth masses
    rstar_rs  = float(getattr(r, "st_rad"))      # Rsun

    err = k218_ref_err * 10.0 ** ((jmag - k218_ref_jmag) / 5.0) * (1.0 / np.sqrt(dur))
    H   = scale_height(teq, rp_re, mp_me)
    depth_err_H = err / ((2 * (rp_re * R_E) * 5.0 * H) / (rstar_rs * R_S) ** 2)
    y_plot = 1.0 / depth_err_H

    rows.append({
        "planet_name": upper_except_last_letter(getattr(r, "match_key", "Target")),
        "jwst_status": _status_label(is_blue=True, is_nirspec=False),
        "published": False,
        "spec_category": "other",
        "is_nirspec": False,
        "x_eq_temp_K": teq,
        "y_snr_per_5H_per_visit": float(y_plot),

        # requested extras
        "radius": rp_re,      # Earth radii
        "mass": mp_me,        # Earth masses
        "jmag": jmag,
        "rstar": rstar_rs,    # Rsun
        "teq": teq,           # K
        "dur": dur,           # hours
    })

# 2) RED/ORANGE (Observed) — prefer values on extended_red_df, else fall back to red_df maps
for r in extended_red_df.itertuples(index=False):
    if not np.isfinite(r.teq) or not np.isfinite(r.depth_err_H):
        continue
    y_plot = 1.0 / float(r.depth_err_H)
    mkey = getattr(r, "match_key", None)

    # pull or fall back
    jmag = float(getattr(r, "jmag")) if hasattr(r, "jmag") and pd.notna(getattr(r, "jmag")) else map_sy_jmag.get(mkey, np.nan)
    rstar_rs = float(getattr(r, "rstar")) if hasattr(r, "rstar") and pd.notna(getattr(r, "rstar")) else map_st_rad.get(mkey, np.nan)
    rp_re = float(getattr(r, "rp")) if hasattr(r, "rp") and pd.notna(getattr(r, "rp")) else (
        map_pl_radj.get(mkey, np.nan) * RJ_TO_RE if pd.notna(map_pl_radj.get(mkey, np.nan)) else np.nan
    )
    mp_me = float(getattr(r, "mp")) if hasattr(r, "mp") and pd.notna(getattr(r, "mp")) else (
        map_pl_massj.get(mkey, np.nan) * MJ_TO_ME if pd.notna(map_pl_massj.get(mkey, np.nan)) else np.nan
    )
    dur = map_trandur.get(mkey, np.nan)

    rows.append({
        "planet_name": upper_except_last_letter(getattr(r, "planet_name", "")),
        "jwst_status": _status_label(is_blue=False, is_nirspec=bool(getattr(r, "is_nirspec", False))),
        "published": bool(getattr(r, "published", False)),
        "spec_category": getattr(r, "spec_category", ""),
        "is_nirspec": bool(getattr(r, "is_nirspec", False)),
        "x_eq_temp_K": float(r.teq),
        "y_snr_per_5H_per_visit": float(y_plot),

        # requested extras
        "radius": rp_re,      # Earth radii
        "mass": mp_me,        # Earth masses
        "jmag": jmag,
        "rstar": rstar_rs,    # Rsun
        "teq": float(r.teq),  # K
        "dur": dur,           # hours
    })

# Assemble + optional rounding for readability
table_df = pd.DataFrame(rows)
for c in ["x_eq_temp_K", "y_snr_per_5H_per_visit", "radius", "mass", "jmag", "rstar", "teq", "dur"]:
    table_df[c] = pd.to_numeric(table_df[c], errors="coerce")

# (optional) pretty rounding
table_out = table_df.copy()
table_out["x_eq_temp_K"]            = table_out["x_eq_temp_K"].round(1)
table_out["y_snr_per_5H_per_visit"] = table_out["y_snr_per_5H_per_visit"].round(4)
table_out["radius"]                 = table_out["radius"].round(3)
table_out["mass"]                   = table_out["mass"].round(3)
table_out["jmag"]                   = table_out["jmag"].round(3)
table_out["rstar"]                  = table_out["rstar"].round(3)
table_out["teq"]                    = table_out["teq"].round(0)
table_out["dur"]                    = table_out["dur"].round(3)

out_path = "niriss_neptune_scales_table.csv"
table_out.to_csv(out_path, index=False)
print(f"\nSaved enriched table to {out_path}\n")
print(table_out.head(20).to_string(index=False))
