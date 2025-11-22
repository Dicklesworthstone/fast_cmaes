use crate::{CmaesState, CovarianceModeKind};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Return a simple version number (major<<16 | minor<<8 | patch).
#[no_mangle]
pub extern "C" fn fastcma_version() -> u32 {
    const VER: &str = env!("CARGO_PKG_VERSION");
    let mut parts = VER.split('.').filter_map(|p| p.parse::<u32>().ok());
    let major = parts.next().unwrap_or(0);
    let minor = parts.next().unwrap_or(0);
    let patch = parts.next().unwrap_or(0);
    (major << 16) | (minor << 8) | patch
}

/// Minimize sphere in C: returns best value; fills xmin buffer if provided.
/// Safety: caller must ensure `xmin` points to at least `dim` f64s when non-null.
#[no_mangle]
pub unsafe extern "C" fn fastcma_sphere(
    dim: usize,
    sigma: f64,
    maxfevals: usize,
    seed: u64,
    xmin: *mut f64,
) -> f64 {
    let x0 = vec![0.6; dim];
    let objective = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // deterministic popsize
    let pop = None;
    let mut es = CmaesState::new_with_seed(
        x0.clone(),
        sigma,
        pop,
        Some(1e-12),
        Some(maxfevals),
        CovarianceModeKind::Full,
        rng.next_u64(),
    );
    while !es.has_terminated() {
        let arx = es.ask();
        let fit: Vec<f64> = arx.iter().map(|x| objective(x)).collect();
        es.tell(arx, fit);
    }
    let (xbest, fbest, _eb, _ce, _it, _xm, _std) = es.result();
    if !xmin.is_null() {
        let count = usize::min(dim, xbest.len());
        for i in 0..count {
            *xmin.add(i) = xbest[i];
        }
    }
    fbest
}
