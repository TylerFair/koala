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
mmw   = 20.0

# --- Plot color constants (human-editable)
COLOR_RED     = "r"
COLOR_BLUE    = "b"
COLOR_PURPLE  = "purple"
COLOR_ORANGE  = "orange"  # For "No NIRSpec – NIRISS/NIRCam only" and unpublished OTHERs

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
        "pl_trandur": float(r.get("pl_trandur", np.nan)),
    }

def scale_height(teq, rp_re, mp_me):
    return (k_B * teq * (rp_re*R_E)**2) / (mmw * m_H * G * (mp_me * M_E))

def depth_err_H_from_err(err, teq, rp_re, mp_me, rstar_rs):
    H = scale_height(teq, rp_re, mp_me)
    return err / ((2 * (rp_re * R_E) * H) / (rstar_rs * R_S)**2)

# NIRSpec data reader: use 4.0–4.3 µm band and Kmag scaling
def grab_planet(infile, pl_name, n_visits=1, R=100, archive=False):
    df1 = pd.read_csv(infile)
    if archive:
        wave, depth, derr = df1.CENTRALWAVELNG.values, df1.PL_TRANDEP.values/100, df1.PL_TRANDEPERR1.values/100
    else:
        wave, depth, derr = df1.wavelength.values, df1.depth.values, df1.depth_err.values
    if R != 40:
        wave, _, depth, derr = bin_at_resolution(wave, depth, derr, 40, "average")

    p = planet_params(red_df, pl_name)
    mask = (wave > 4.0) & (wave < 4.3)
    err_band = np.nanmean(derr[mask]) * (1/np.sqrt(n_visits))

    H = scale_height(p["teq"], p["rad"], p["mass"])
    N_H = (err_band * (p["rstar"] * R_S)**2) / (2 * (p["rad"] * R_E) * H)
    depth_err_H = depth_err_H_from_err(err_band, p["teq"], p["rad"], p["mass"], p["rstar"])

    return p["teq"], depth_err_H, p["rad"], p["mass"], p["rstar"], p["Kmag"], err_band, N_H

def manual_planet(error_ppm, pl_name, n_visits=1):
    p  = planet_params(red_df, pl_name)
    err = error_ppm*1e-6 * (1/np.sqrt(n_visits))
    depth_err_H = depth_err_H_from_err(err, p["teq"], p["rad"], p["mass"], p["rstar"])
    H = scale_height(p["teq"], p["rad"], p["mass"])
    N_H = (err * (p["rstar"] * R_S)**2) / (2 * (p["rad"] * R_E) * H)
    return p["teq"], depth_err_H, p["rad"], p["mass"], p["rstar"], p["Kmag"], err, N_H

# --------------------
# Load + prefilter (NIRSpec/Kmag)
# --------------------
targets = pd.read_csv("subneptune_target_list_with_nasa_ps_filled.csv")
targets = targets[(targets["snr"] >= 1.0) & (targets["kmag"] >= 7.8)]

df = pd.read_csv("03_trexolists_extended.csv")
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

# targets match_key
targets = ensure_match_key(
    targets.assign(match_key=(planet_key(targets["pl_name"]) if "pl_name" in targets.columns and not targets["pl_name"].isna().all()
                              else planet_key(targets.get("star","").astype(str))))
)

# red_df (observed/approved set)
red_df = ensure_match_key(keep_target.copy())

# which targets are not in red set
blue_targets = targets[~targets["match_key"].isin(set(red_df["match_key"]))].copy()

# normalized densities (if needed later)
blue_targets["rho_norm"] = normalized_density(blue_targets, "pl_bmasse", "pl_dens", earth_rho_fn)
red_df["pl_masse_from_j"] = (red_df["pl_massj"].to_numpy(float) * const.M_jup / const.M_earth).value
red_df["rho_norm"] = normalized_density(red_df, "pl_masse_from_j", "pl_dens_cgs", earth_rho_fn)

# --------------------
# Build NIRSpec planet set (grab spectra / manual)
# --------------------
#toi421b   = grab_planet('spectra/TOI-421/SPECTRUM/TOI421b_NIRSPEC_G395M_nrs1_R100.csv', "TOI-421 b")
toi421b   = manual_planet(25, "TOI-421 b", n_visits=1)

hatp11b   = grab_planet('spectra/HAT-P-11/SPECTRUM/HATP11b_NIRSPEC_G395H_nrs2_R200.csv', "HAT-P-11 b")
toi178d   = grab_planet('spectra/TOI-178/NRS1_FREELD_LINEAR/TOI178d_NIRSPEC_G395M_nrs1_R100.csv', "TOI-178 d")
toi1468c  = grab_planet('spectra/TOI-1468/NRS2_FREELD_LINEAR/TOI1468c_NIRSPEC_G395H_nrs2_R100.csv', "TOI-1468 c")
ltt3780c  = grab_planet('spectra/LTT-3780/NRS2_FREELD_LINEAR/LTT3780c_NIRSPEC_G395H_nrs2_R50.csv', "LTT 3780 c")
toi1231b  = grab_planet('spectra/TOI-1231/NRS2_FREELD_LINEAR/TOI1231b_NIRSPEC_G395H_nrs2_R100.csv', "TOI-1231 b")
toi1130  = grab_planet('spectra/TOI-1130/NRS2_FIXEDLD_LINEAR/TOI1130b_NIRSPEC_G395H_nrs2_R200.csv', "TOI-1130 b")
toi270d   = manual_planet(40, "TOI-270 d", n_visits=2)

# Published (archive) NIRSpec/NIRCam references
k218b_p   = grab_planet('K2-18_nirspec_paper.csv',       "K2-18 b",   archive=True, n_visits=1)
lhs1140b_p= grab_planet('LHS-1140_nirspec_paper.csv',    "LHS 1140 b",archive=True)
toi836b_p = grab_planet('TOI-836_nirspec_paper.csv',     "TOI-836 b", archive=True)
toi776b_p = grab_planet('TOI-776_nirspec_paper.csv',     "TOI-776 b", archive=True)
toi776c_p = grab_planet('TOI-776c_nirspec_paper.csv',    "TOI-776 c", archive=True)
gj1214b_p = grab_planet('GJ-1214b_nirspec_paper.csv',    "GJ 1214 b", archive=True)
toi836c_p = grab_planet('TOI-836c_nirspec_paper.csv',    "TOI-836 c", archive=True)
gj3470b_p = grab_planet('GJ-3470b_nircam_paper.csv',     "GJ 3470 b", archive=True, n_visits=2)

published_planets = [
    ("TOI-421 b", toi421b),
    ("K2-18 b", k218b_p),
    ("LHS 1140 b", lhs1140b_p),
    ("TOI-836 b", toi836b_p),
    ("TOI-776 b", toi776b_p),
    ("TOI-776 c", toi776c_p),
    ("GJ 1214 b", gj1214b_p),
    ("TOI-836 c", toi836c_p),
    ("GJ 3470 b", gj3470b_p),
    ("TOI-270 d", toi270d),
]
unpublished_planets = [
    ("HAT-P-11 b", hatp11b),
    ("TOI-178 d",  toi178d),
    ("TOI-1468 c", toi1468c),
    ("LTT 3780 c", ltt3780c),
    ("TOI-1231 b", toi1231b),
    ("TOI-1130 b", toi1130),
    #("LP 791-18 c", lp79118c),
]

manual_rows = []
def _add_manual(name, data, published_flag, is_nirspec=True):
    manual_rows.append({
        "planet_name": name,
        "match_key": planet_key(pd.Series([name])).iloc[0],
        "teq": data[0], "depth_err_H": data[1], "rp": data[2], "mp": data[3],
        "rstar": data[4], "kmag": data[5],
        "published": bool(published_flag),
        "spec_category": "other",   # will be overwritten below
        "is_nirspec": bool(is_nirspec),
    })

for name, data in published_planets:
    _add_manual(name, data, True,  is_nirspec=True)
for name, data in unpublished_planets:
    _add_manual(name, data, False, is_nirspec=True)

manual_red_df = pd.DataFrame(manual_rows)

# --------------------
# Spectrum categories and published flags from full df
# --------------------
df_for_symbols = ensure_match_key(df.copy())
spec_series = df_for_symbols["spectrum"].astype(str) if "spectrum" in df_for_symbols.columns else pd.Series(index=df_for_symbols.index, dtype=str)
is_clear = spec_series.str.contains(r"\bclear\b", case=False, na=False)
is_flat  = spec_series.str.contains(r"\bflat\b",  case=False, na=False)
agg = (df_for_symbols.assign(__clear=is_clear, __flat=is_flat)
       .groupby("match_key", dropna=False)[["__clear","__flat"]].any().reset_index())

cat_map = {mk: ("clear" if c else "flat" if f else "other")
           for mk, c, f in agg[["match_key","__clear","__flat"]].itertuples(index=False)}
manual_red_df["spec_category"] = manual_red_df["match_key"].map(cat_map).fillna("other")

pub_agg = (df_for_symbols.assign(__pub=df_for_symbols.get("published","").astype(str)
                                 .str.strip().str.lower().isin({"yes","y","true","published"}))
           .groupby("match_key", dropna=False)["__pub"].any())
manual_red_df["published"] = manual_red_df["match_key"].map(pub_agg).fillna(False) | manual_red_df["published"].fillna(False)

# --------------------
# Unsure + unpublished candidates to extend red set
# --------------------
is_unsure = spec_series.str.contains(r"\bunsure\b", case=False, na=False)
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

for col in ["pl_Teq_K","sy_kmag","pl_trandur","pl_radj","pl_massj","st_rad"]:
    unsure_unpub[col] = pd.to_numeric(unsure_unpub.get(col), errors="coerce")

# Reference scaling values from K2-18 b (NIRSpec)
k218_ref_err   = float(k218b_p[6])
k218_ref_kmag  = float(k218b_p[5])
k218_ref_NH    = float(k218b_p[7])

unsure_unpub = unsure_unpub.assign(
    planet_name = unsure_unpub.apply(friendly_name, axis=1),
    teq   = unsure_unpub["pl_Teq_K"],
    kmag  = unsure_unpub["sy_kmag"],
    rp    = unsure_unpub["pl_radj"] * R_J_R_E,
    mp    = unsure_unpub["pl_massj"] * (const.M_jup/const.M_earth).value,
    rstar = unsure_unpub["st_rad"],
    pl_trandur = unsure_unpub["pl_trandur"],
    published=False,
    spec_category="other",
    is_nirspec=True,  # by default, these are not yet NIRSpec-confirmed
)

H_u = scale_height(unsure_unpub["teq"], unsure_unpub["rp"], unsure_unpub["mp"])
err_band_u = (k218_ref_err * 10.0 ** ((unsure_unpub["kmag"] - k218_ref_kmag)/5)) * (1/np.sqrt(unsure_unpub["pl_trandur"]))
unsure_unpub["depth_err_H"] = ( err_band_u /
    ((2 * (unsure_unpub["rp"] * R_E) * H_u) / (unsure_unpub["rstar"] * R_S)**2) )

cols = ["planet_name","match_key","teq","depth_err_H","rp","mp","rstar","kmag","published","spec_category","is_nirspec"]

# --------------------
# Combine sets + manual category overrides
# --------------------
extended_red_df = pd.concat([manual_red_df, unsure_unpub[cols]], ignore_index=True, sort=False)

override_flat = planet_key(pd.Series(['TOI-1468 c', 'LTT 3780 c', 'TOI-178 d'])).tolist()
extended_red_df.loc[extended_red_df["match_key"].isin(override_flat), "spec_category"] = "flat"

# --------------------
# PLOTTING (NIRSpec style, inverted: plot 1 / depth_err_H = signal)
# --------------------
plt.figure(figsize=(12, 8))

# Blue points (potential targets)
blue_sc = None
for r in blue_targets.itertuples(index=False):
    err = k218_ref_err * 10 ** ((getattr(r, "kmag") - k218_ref_kmag)/5) * (1/np.sqrt(getattr(r, "pl_trandur")))
    H   = scale_height(getattr(r, "scale_temp"), getattr(r, "pl_rade"), getattr(r, "pl_bmasse"))
    depth_err_H = err / ((2 * (getattr(r, "pl_rade") * R_E) * H) / (getattr(r, "st_rad") * R_S)**2)

    if blue_sc is None:
        blue_sc = plt.scatter(getattr(r, "scale_temp"), 1/depth_err_H, c=COLOR_BLUE, alpha=0.7, s=30, label="Potential Targets")
    else:
        plt.scatter(getattr(r, "scale_temp"), 1/depth_err_H, c=COLOR_BLUE, alpha=0.7, s=30)

    plt.annotate(upper_except_last_letter(getattr(r, "match_key", "Target")),
                 (getattr(r, "scale_temp"), 1/depth_err_H),
                 color="navy", textcoords="offset points", xytext=(3,3),
                 ha="left", va="bottom", fontsize=8,
                 zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Red/Orange points by category
x = extended_red_df["teq"].to_numpy(float)
y = extended_red_df["depth_err_H"].to_numpy(float)
valid = np.isfinite(x) & np.isfinite(y)
spec = extended_red_df["spec_category"].astype(str).str.lower()
pub  = extended_red_df["published"].to_numpy(bool)
is_nirspec = extended_red_df["is_nirspec"].fillna(False).to_numpy(bool)

mask_clear = (spec == "clear") & valid
mask_flat  = (spec == "flat")  & valid
mask_other = (spec == "other") & valid

# Invert color rule relative to NIRISS code:
# ORANGE = NOT NIRSpec (i.e., NIRISS/NIRCam only) OR unpublished "other"
is_orange = (~is_nirspec) #| ((~pub) & mask_other)
is_red    = ~is_orange

# CLEAR
plt.scatter(x[mask_clear & is_red],    1/y[mask_clear & is_red],    c=COLOR_RED,    marker="*", s=150, zorder=4)
plt.scatter(x[mask_clear & is_orange], 1/y[mask_clear & is_orange], c=COLOR_ORANGE, marker="*", s=150, zorder=4)

# FLAT
plt.scatter(x[mask_flat & is_red],     1/y[mask_flat & is_red],     c=COLOR_RED,    marker="_", s=200, linewidths=2, zorder=5)
plt.scatter(x[mask_flat & is_orange],  1/y[mask_flat & is_orange],  c=COLOR_ORANGE, marker="_", s=200, linewidths=2, zorder=5)

# OTHER
plt.scatter(x[mask_other & is_red],    1/y[mask_other & is_red],    c=COLOR_RED,    marker="o", s=80,  zorder=4)
plt.scatter(x[mask_other & is_orange], 1/y[mask_other & is_orange], c=COLOR_ORANGE, marker="o", s=80,  zorder=4)

# Published overlays
plt.scatter(x[mask_clear & pub], 1/y[mask_clear & pub], facecolors="none", edgecolors=COLOR_PURPLE,
            marker="*", s=220, linewidths=2, zorder=6)
plt.scatter(x[mask_other & pub], 1/y[mask_other & pub], facecolors="none", edgecolors=COLOR_PURPLE,
            marker="o", s=110, linewidths=2, zorder=6)
plt.scatter(x[mask_flat & pub],  1/y[mask_flat & pub],  c=COLOR_PURPLE, marker="_", s=500, linewidths=2, zorder=3)

# Annotations (orange first beneath, red above)
text_kw = dict(textcoords="offset points", xytext=(3,3), ha="left", va="bottom",
               fontsize=8, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

for _, row in extended_red_df.loc[valid & is_orange].iterrows():
    plt.annotate(
        upper_except_last_letter(row["planet_name"]),
        (row["teq"], 1/row["depth_err_H"]),
        color="darkorange",
        zorder=6,
        **text_kw
    )

for _, row in extended_red_df.loc[valid & is_red].iterrows():
    plt.annotate(
        upper_except_last_letter(row["planet_name"]),
        (row["teq"], 1/row["depth_err_H"]),
        color="darkred",
        zorder=8,
        **text_kw
    )

# Legend (NIRSpec-flipped orange meaning)
ax = plt.gca()
handles = [
    *( [blue_sc] if blue_sc is not None else [] ),
    Line2D([], [], marker="*", color=COLOR_RED,    linestyle="None", markersize=8,  label="NIRSpec - Features"),
    Line2D([], [], marker="_", color=COLOR_RED,    linestyle="None", markersize=14, label="NIRSpec - Flat"),
    Line2D([], [], marker="o", color=COLOR_RED,    linestyle="None", markersize=6,  label="NIRSpec - Unsure/Unknown"),
  #  Line2D([], [], marker="o", color=COLOR_ORANGE, linestyle="None", markersize=6,  label="No NIRSpec/NIRCam - NIRISS Only"),
    Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor=COLOR_PURPLE, markersize=6, label="Published (purple outline)"),
]
labels = [h.get_label() for h in handles if h is not None]
ax.legend([h for h in handles if h is not None], labels, frameon=True, title="Legend", scatterpoints=1)

plt.xlabel("Equilibrium Temperature (K)")
plt.ylabel("Signal of a 1 Scale Height Feature @ μ=20 amu (NIRSpec 4.0–4.3µm)")
plt.yscale("log")
plt.tight_layout()
plt.savefig("nirspec_neptune_scales.png", dpi=300, bbox_inches='tight')
plt.show()


def _load_spectrum(infile, archive=False, R_plot=100, res=300):
    """Return wavelength [µm], depth (fraction), depth_err (fraction), binned to R_plot."""
    df = pd.read_csv(infile)
    if archive:
        wave = df["CENTRALWAVELNG"].to_numpy(float)
        depth = (df["PL_TRANDEP"].to_numpy(float)) / 100.0
        derr  = (df["PL_TRANDEPERR1"].to_numpy(float)) / 100.0
    else:
        wave = df["wavelength"].to_numpy(float)
        depth = df["depth"].to_numpy(float)
        derr  = df["depth_err"].to_numpy(float)
    if res != R_plot:
    
        bw, _, bd, be = bin_at_resolution(wave, depth, derr, R_plot, "average")
        return bw, bd, be
    
    return wave, depth, derr

def _compute_A_H(waves, depths, teq, rp_re, mp_me, rstar_rs):
    """Compute the A_H normalization used inside scale_height_normalized_spectrum."""
    H = scale_height(teq, rp_re, mp_me)
    mask_H2O = (waves >= 3) & (waves <= 5)
    mask_J   = (waves >= 4.25) & (waves <= 4.5)
    print(np.nanmean(depths[mask_H2O]), np.nanmean(depths[mask_J]))
    med_depths = np.nanmean(depths[mask_H2O]) - np.nanmean(depths[mask_J])

    return med_depths / ((2 * (rp_re * R_E) * H) / (rstar_rs * R_S)**2)

spectra_catalog = [
    # name, path, archive_flag
    ("HAT-P-11 b", 'spectra/HAT-P-11/SPECTRUM/HATP11b_NIRSPEC_G395H_nrs2_R200.csv', False, 200),
    ("TOI-178 d",  'spectra/TOI-178/NRS1_FREELD_LINEAR/TOI178d_NIRSPEC_G395M_nrs1_R100.csv', False, 100),
    ("TOI-1468 c", 'spectra/TOI-1468/NRS2_FREELD_LINEAR/TOI1468c_NIRSPEC_G395H_nrs2_R100.csv', False, 100),
    ("LTT 3780 c", 'spectra/LTT-3780/NRS2_FREELD_LINEAR/LTT3780c_NIRSPEC_G395H_nrs2_R50.csv', False, 50),
    ("TOI-1231 b", "spectra/TOI-1231/NRS2_FREELD_LINEAR/TOI1231b_NIRSPEC_G395H_nrs2_R100.csv", False, 100),
    ("TOI-1130 b", "spectra/TOI-1130/NRS2_FIXEDLD_LINEAR/TOI1130b_NIRSPEC_G395H_nrs2_R200.csv", False, 100),
    ("K2-18 b",    "K2-18_nirspec_paper.csv", True, 300),
   # ("GJ 9827 d",  "GJ-9827d_niriss_paper.csv", True, 300),
    ("LHS-1140 b", "LHS-1140_nirspec_paper.csv", True, 300),
    ("TOI-836 b",  "TOI-836_nirspec_paper.csv", True, 300),
    ("TOI-776 b",  "TOI-776_nirspec_paper.csv", True, 300),
    ("TOI-776 c",  "TOI-776c_nirspec_paper.csv", True, 300),
    ("GJ 1214 b",  "GJ-1214b_nirspec_paper.csv", True, 300),
    ("TOI-836 c",  "TOI-836c_nirspec_paper.csv", True, 300),
    ("GJ 3470 b",  "GJ-3470b_nircam_paper.csv", True, 300),
]



R_plot = 50          # target plotting resolution
OFFSET= 4e-6

plt.figure(figsize=(12, 7))
ax = plt.gca()

N = len(spectra_catalog)
cmap = plt.get_cmap("turbo")  # or "Spectral", "tab20"
colors = [cmap(i/(N-1 if N>1 else 1)) for i in range(N)]

handles = []
labels  = []

for i, (name, path, is_archive, res) in enumerate(spectra_catalog):
    try:
        waves, depths, errs = _load_spectrum(path, archive=is_archive, R_plot=R_plot, res=res)
        if name == 'GJ 9827 d':
            print(len(waves), waves)
            print(len(depths), depths)
        p = planet_params(red_df, name) 
        # Normalize to A_H 
        A_H = _compute_A_H(waves, depths, p["teq"], p["rad"], p["mass"], p["rstar"])
        Jmask   = (waves >= 4.25) & (waves <= 4.5)
        medJ  = np.nanmean(depths[Jmask])
     

        y_AH    =  np.abs((depths - medJ)  / A_H)

        yerr_AH = np.abs(errs / A_H)

        yoffs = OFFSET * i 
        col = colors[i]

        # line + light markers
        ax.errorbar(waves, y_AH + yoffs, yerr=yerr_AH, ms=4, fmt='o', ls='', color=col, alpha=0.5, zorder=4)

        labels.append(f"{upper_except_last_letter(name)}  (+{i*OFFSET:.0f} A_H)")
        # annotate on the right
        ax.text(5.25, yoffs, upper_except_last_letter(name),
                va="center", fontsize=9, color=col,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    except Exception as e:
        print(f"[WARN] Skipped {name}: {e}")

ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Transit Depth [H @ 20 MMW]")
ax.set_xlim(3, 5.3)
ax.set_ylim(-2.5e-5, 7.5e-5)
plt.tight_layout()
plt.savefig("nirspec_all_spectra_AH_offsets_20amu.png", dpi=300, bbox_inches="tight")
plt.show()

