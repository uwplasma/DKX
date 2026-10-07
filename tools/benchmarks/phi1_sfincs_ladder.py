import json, sys, time, numpy as np
from dkx.phi1 import solve_phi1
from dkx.run import profile_moments_from_operator
out = {}
for d in sys.argv[1:]:
    t = time.time()
    try:
        r = solve_phi1(f"{d}/input.namelist", tol=1e-11)
        m = profile_moments_from_operator(r.operator, r.x)
        phi = np.asarray(r.phi1_hat)
        out[d] = dict(flux=np.asarray(m["particleFlux_vm_psiHat"]).tolist(), phi_rms=float(np.sqrt(np.mean(phi**2))),
                      phi_max=float(np.max(np.abs(phi))), res=r.residual_norm, conv=bool(r.converged), s=time.time() - t)
    except Exception as e:
        out[d] = dict(error=repr(e)[:300], s=time.time() - t)
    print(d, out[d], flush=True)
    json.dump(out, open(f"./dkx_{'_'.join(sys.argv[1:])}.json", "w"))
