"""
Compare a solver output file with the expected values for the homogeneous test.

    python3 check_output.py modulus_output_homogeneous_tif_factor1.0_x.txt 37 44

arguments: output file, input K, input G (same units as in the input file).
For a homogeneous solid the affine boundary conditions give the exact solution,
so K_eff and G_eff must equal K and G (to solver tolerance, about 1e-5 relative).
"""
import re
import sys

fname, K, G = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
text = open(fname).read()
Keff = float(re.search(r"^K_eff:\s*([-\d.eE+]+)", text, re.M).group(1))
Geff = float(re.search(r"^G_eff:\s*([-\d.eE+]+)", text, re.M).group(1))
ok = True
for name, got, want in (("K_eff", Keff, K), ("G_eff", Geff, G)):
    err = abs(got - want) / want
    flag = "OK" if err < 1e-3 else "MISMATCH"
    ok &= err < 1e-3
    print(f"{name}: got {got:.4f}, expected {want:.4f}, relative error {err:.1e}  {flag}")
sys.exit(0 if ok else 1)
