use fastcma::CovarianceModeKind;
use fastcma::test_utils::{run_seeded_mode, run_seeded_mode_noise};
use std::time::Instant;

fn ill_cond_ellipsoid(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0;
    for (i, &v) in x.iter().enumerate() {
        let exp = 6.0 * (i as f64) / ((n - 1) as f64);
        let coeff = 10f64.powf(exp);
        s += coeff * v * v;
    }
    s
}

fn infeasible_penalty(x: &[f64]) -> f64 {
    let base: f64 = x.iter().map(|v| v * v).sum();
    let g = x[0] + x[1] - 0.5; // should be <= 0
    let penalty = (g.max(0.0)).powi(2) * 100.0;
    base + penalty
}

#[test]
fn stability_ellipsoid_high_condition() {
    let start = Instant::now();
    let f = run_seeded_mode(
        vec![0.6; 10],
        0.3,
        80_000,
        1e-10,
        4242,
        CovarianceModeKind::Full,
        ill_cond_ellipsoid,
    );
    assert!(f < 1e-4, "ill-conditioned ellipsoid not solved: {f}");
    assert!(
        start.elapsed().as_millis() < 40_000,
        "ellipsoid timing regression"
    );
}

#[test]
fn stability_noisy_sphere_penalty_path() {
    let start = Instant::now();
    let f = run_seeded_mode_noise(
        vec![0.4, 0.4],
        0.5,
        20_000,
        1e-6,
        999,
        CovarianceModeKind::Full,
        infeasible_penalty,
    );
    assert!(f < 0.05, "penalty path failed under noise: {f}");
    assert!(
        start.elapsed().as_millis() < 500,
        "penalty timing regression"
    );
}
