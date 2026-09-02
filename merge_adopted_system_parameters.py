#!/usr/bin/env python3
"""Merge the adopted retrieval table with representative white-light geometry."""

from pathlib import Path
import pandas as pd


# Active (uncommented) rows supplied for the adopted retrieval table.  Geometry
# is inserted after orbital period from the first configured dataset per planet.
ROWS = [
r"WASP-96b & \paramerr{5781}{69}{69} & 1.055 & \paramerr{4.48}{0.11}{0.11} & \paramerr{0.43}{0.05}{0.05} & 3.425257 & \paramerr{0.11906}{0.00008}{0.00008} & 1.200 & \paramerr{0.470}{0.036}{0.034} & 971 & \texttt{S01} & \texttt{P01}",
r"WASP-94 Ab & \paramerr{6217}{19}{19} & 1.620 & \paramerr{4.27}{0.04}{0.04} & \paramerr{0.37}{0.01}{0.01} & 3.950200 & \paramerr{0.10601}{0.00004}{0.00004} & 1.580 & \paramerr{0.456}{0.034}{0.036} & 1604 & \texttt{S02} & \texttt{P02}",
r"WASP-69b & \paramerr{4802}{100}{100} & 0.801 & \paramerr{4.15}{0.26}{0.26} & \paramerr{0.29}{0.04}{0.04} & 3.868139 & \paramerr{0.12850}{0.00002}{0.00002} & 1.000 & \paramerr{0.260}{0.020}{0.020} & 971 & \texttt{S04} & \texttt{P04}",
r"WASP-63b & \paramerr{5676}{40}{40} & 1.790 & \paramerr{4.12}{0.06}{0.06} & \paramerr{0.26}{0.03}{0.03} & 4.378082 & \paramerr{0.07826}{0.00004}{0.00004} & 1.410 & \paramerr{0.339}{0.030}{0.030} & 1470 & \texttt{S05} & \texttt{P05}",
r"WASP-52b & \paramerr{5073}{63}{63} & 0.860 & \paramerr{4.36}{0.15}{0.15} & \paramerr{0.17}{0.03}{0.03} & 1.749781 & \paramerr{0.16624}{0.00005}{0.00005} & 1.270 & \paramerr{0.459}{0.021}{0.022} & 1315 & \texttt{S07} & \texttt{P07}",
r"WASP-39b & \paramerr{5509}{28}{28} & 1.013 & \paramerr{4.22}{0.07}{0.07} & \paramerr{0.04}{0.02}{0.02} & 4.055280 & \paramerr{0.14594}{0.00003}{0.00003} & 1.279 & \paramerr{0.275}{0.043}{0.042} & 1166 & \texttt{S08} & \texttt{P08}",
r"WASP-17b & \paramerr{6793}{52}{52} & 1.570 & \paramerr{4.59}{0.04}{0.04} & \paramerr{-0.01}{0.03}{0.03} & 3.735485 & \paramerr{0.12463}{0.00004}{0.00004} & 1.870 & \paramerr{0.512}{0.037}{0.037} & 1750 & \texttt{S10} & \texttt{P10}",
r"WASP-166b & \paramerr{6142}{22}{22} & 1.250 & \paramerr{4.45}{0.04}{0.04} & \paramerr{0.25}{0.02}{0.02} & 5.443542 & \paramerr{0.05380}{0.00004}{0.00004} & 0.630 & \paramerr{0.101}{0.005}{0.005} & 1270 & \texttt{S11} & \texttt{P11}",
r"WASP-127b & \paramerr{5832}{14}{14} & 1.333 & \paramerr{4.31}{0.03}{0.03} & \paramerr{-0.18}{0.01}{0.01} & 4.178065 & \paramerr{0.10008}{0.00005}{0.00005} & 1.024 & \paramerr{0.165}{0.017}{0.021} & 1401 & \texttt{S12} & \texttt{P12}",
r"WASP-121b & \paramerr{6827}{72}{72} & 1.460 & \paramerr{4.64}{0.06}{0.06} & \paramerr{0.46}{0.05}{0.05} & 1.274925 & \paramerr{0.12255}{0.00005}{0.00005} & 1.740 & \paramerr{1.170}{0.043}{0.043} & 2409 & \texttt{S13} & \texttt{P13}",
r"WASP-107b & \paramerr{4488}{227}{227} & 0.670 & \paramerr{4.15}{0.78}{0.78} & \paramerr{-0.02}{0.13}{0.13} & 5.721488 & \paramerr{0.14384}{0.00004}{0.00004} & 0.790 & \paramerr{0.096}{0.005}{0.005} & 672 & \texttt{S14} & \texttt{P14}",
r"NGTS-2b & \paramerr{6945}{140}{140} & 1.710 & \paramerr{4.98}{0.10}{0.10} & \paramerr{0.01}{0.08}{0.08} & 4.511123 & \paramerr{0.09932}{0.00004}{0.00004} & 1.595 & \paramerr{0.640}{0.120}{0.130} & 1468 & \texttt{S18} & \texttt{P18}",
r"Kepler-12b & \paramerr{5926}{60}{60} & 1.480 & \paramerr{4.01}{0.10}{0.10} & \paramerr{0.03}{0.04}{0.04} & 4.437963 & \paramerr{0.11911}{0.00006}{0.00006} & 1.754 & \paramerr{0.433}{0.040}{0.042} & 1480 & \texttt{S22} & \texttt{P22}",
r"KELT-7b & \paramerr{6789}{50}{50} & 1.770 & \paramerr{4.23}{0.10}{0.10} & \paramerr{0.14}{0.08}{0.08} & 2.734766 & \paramerr{0.09026}{0.00002}{0.00002} & 1.600 & \paramerr{1.280}{0.180}{0.180} & 2048 & \texttt{S23} & \texttt{P23}",
r"HAT-P-65b & \paramerr{5835}{51}{51} & 1.550 & \paramerr{4.18}{0.10}{0.10} & \paramerr{0.10}{0.08}{0.08} & 2.605448 & \paramerr{0.10270}{0.00007}{0.00007} & 1.611 & \paramerr{0.554}{0.091}{0.092} & 1818 & \texttt{S25} & \texttt{P25}",
r"HAT-P-30b & \paramerr{6321}{30}{30} & 1.340 & \paramerr{4.45}{0.04}{0.04} & \paramerr{0.10}{0.02}{0.02} & 2.810601 & \paramerr{0.10931}{0.00028}{0.00028} & 1.440 & \paramerr{0.746}{0.021}{0.020} & 1630 & \texttt{S26} & \texttt{P26}",
r"HAT-P-18b & \paramerr{4710}{238}{238} & 0.749 & \paramerr{4.17}{0.58}{0.58} & \paramerr{0.06}{0.08}{0.08} & 5.508029 & \paramerr{0.13705}{0.00008}{0.00008} & 0.999 & \paramerr{0.197}{0.013}{0.013} & 839 & \texttt{S27} & \texttt{P27}",
r"HAT-P-12b & \paramerr{4779}{108}{108} & 0.701 & \paramerr{4.22}{0.26}{0.26} & \paramerr{-0.28}{0.04}{0.04} & 3.213058 & \paramerr{0.13730}{0.00009}{0.00009} & 0.949 & \paramerr{0.208}{0.009}{0.010} & 975 & \texttt{S29} & \texttt{P29}",
r"HAT-P-11b & \paramerr{4692}{97}{97} & 0.760 & \paramerr{3.99}{0.30}{0.30} & \paramerr{0.12}{0.05}{0.05} & 4.887802 & \paramerr{0.05854}{0.00005}{0.00005} & 0.447 & \paramerr{0.079}{0.005}{0.005} & 847 & \texttt{S30} & \texttt{P30}",
r"HAT-P-26b & \paramerr{5001}{61}{61} & 0.916 & \paramerr{4.21}{0.15}{0.15} & \paramerr{0.01}{0.04}{0.04} & 4.234500 & \paramerr{0.07226}{0.00009}{0.00009} & 0.652 & \paramerr{0.058}{0.007}{0.007} & 1080 & \texttt{S33} & \texttt{P33}",
]


def perr(row, key, centre_digits, error_digits=2):
    return (rf"\paramerr{{{float(row[key]):.{centre_digits}f}}}"
            rf"{{{float(row[key + '_err_low']):.{error_digits}g}}}"
            rf"{{{float(row[key + '_err_high']):.{error_digits}g}}}")


def canonical(name):
    return name.replace(" Ab", "").removesuffix("b")


def main():
    root = Path(__file__).resolve().parent
    fits = pd.read_csv(root / "supplementary_lightcurve_products/white_light_parameters.csv")
    fits = fits.drop_duplicates("planet").set_index("planet")
    output = [
        r"\providecommand{\paramerr}[3]{\ensuremath{#1_{-#2}^{+#3}}}", "",
        r"\begin{table*}[p]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Adopted stellar and planetary parameters for retrievals and representative white-light transit geometry. The fitted $T_{14}$, $b$, and transformed $a/R_\star$ values are from the first configured dataset for each planet; the complete detector- and visit-specific results are given in Supplementary Table~\ref{tab:white-light-parameters-all}.}",
        r"\label{tab:system-parameters}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccccccc}", r"\hline",
        r"Planet & $T_{\rm eff}$ & $R_\star$ & $\log g_\star$ & $[{\rm Fe/H}]$ & $P$ & $T_{14}$ & $b$ & $a/R_\star$ & $R_p/R_\star$ & $R_p$ & $M_p$ & $T_{\rm eq}$ & Stellar & Planet \\",
        r" & (K) & ($R_\odot$) & (cgs) & (dex) & (days) & (days) & & & & ($R_{\rm J}$) & ($M_{\rm J}$) & (K) & Ref. & Ref. \\",
        r"\hline",
    ]
    for raw in ROWS:
        fields = [x.strip() for x in raw.split("&")]
        fit = fits.loc[canonical(fields[0])]
        geometry = [perr(fit, "duration", 6), perr(fit, "b", 4), perr(fit, "a_rs", 4)]
        output.append(" & ".join(fields[:6] + geometry + fields[6:]) + r" \\")
    output += [r"\hline", r"\end{tabular}%", r"}", r"\end{table*}"]
    destination = root / "supplementary_lightcurve_products/combined_system_parameters.tex"
    destination.write_text("\n".join(output) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
