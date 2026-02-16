use fastcma::test_utils::{
    run_ipop_bipop_parallel, run_multiseed, run_seeded, run_seeded_mode, run_seeded_mode_noise,
};
use fastcma::{CovarianceModeKind, test_utils::augmented_lagrangian_penalty_raw};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use std::cell::RefCell;
use std::time::Instant;

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
            .sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq = x.iter().map(|v| v * v).sum::<f64>();
    let sum_cos = x
        .iter()
        .map(|v| (2.0 * std::f64::consts::PI * v).cos())
        .sum::<f64>();

    let term1 = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp();
    let term2 = -((sum_cos / n).exp());
    term1 + term2 + 20.0 + std::f64::consts::E
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let xi = x[i];
        let xnext = x[i + 1];
        s += 100.0 * (xi * xi - xnext).powi(2) + (xi - 1.0).powi(2);
    }
    s
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

#[test]
fn sphere_reaches_near_zero() {
    let fbest = run_seeded(vec![0.8; 6], 0.4, 4000, 1e-12, sphere);
    assert!(fbest < 1e-8, "sphere optimum not reached: {fbest}");
}

#[test]
fn rosenbrock_2d_reaches_minimum() {
    let fbest = run_seeded(vec![-1.2, 1.0], 0.5, 20000, 1e-10, rosenbrock);
    assert!(fbest < 1e-4, "rosenbrock optimum not reached: {fbest}");
}

#[test]
fn rastrigin_4d_converges() {
    let start = Instant::now();
    // Rastrigin is multi-modal; explore a small seed set and pick best.
    let best = run_multiseed(
        vec![0.2; 4],
        0.15,
        200_000,
        1e-12,
        &[1337, 7, 99],
        CovarianceModeKind::Full,
        &rastrigin,
    );
    assert!(best < 0.5, "rastrigin optimum not reached: {best}");
    assert!(
        start.elapsed().as_millis() < 1_500,
        "rastrigin timing regression"
    );
}

#[test]
fn rastrigin_bipop_parallel_restarts() {
    let start = Instant::now();
    let fbest = run_ipop_bipop_parallel(
        vec![0.3; 6],
        0.35,
        30_000,
        1e-12,
        3,
        3,
        2025,
        CovarianceModeKind::Full,
        &rastrigin,
    );
    assert!(fbest < 1.5, "bipop parallel restarts stalled: {fbest}");
    assert!(
        start.elapsed().as_millis() < 1_500,
        "bipop timing regression"
    );
}

#[test]
fn augmented_lagrangian_penalty_basic() {
    let g = vec![0.1, -0.2];
    let lambda = vec![1.0, 0.5];
    let rho = 10.0;
    let pen = augmented_lagrangian_penalty_raw(&g, &lambda, rho);
    let expected = 1.0 * 0.1 + 0.5 * 0.0 + 0.5 * rho * 0.1 * 0.1;
    assert!((pen - expected).abs() < 1e-9);
}

#[test]
fn noisy_sphere_noise_handling() {
    let start = Instant::now();
    let rng = RefCell::new(StdRng::seed_from_u64(12345));
    let noise_sphere = |x: &[f64]| {
        let noise: f64 = rng.borrow_mut().sample::<f64, _>(StandardNormal) * 0.05;
        let f = x.iter().map(|v| v * v).sum::<f64>();
        f + noise
    };
    let fbest = run_seeded_mode_noise(
        vec![0.8; 6],
        0.5,
        20_000,
        1e-6,
        2025,
        CovarianceModeKind::Full,
        noise_sphere,
    );
    assert!(fbest < 0.05, "noise handling failed to converge: {fbest}");
    assert!(
        start.elapsed().as_millis() < 800,
        "noisy sphere timing regression"
    );
}

fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|v| v * (v.abs().sqrt()).sin()).sum();
    418.9829 * n - sum
}

fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / 4000.0;
    let prod_cos: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| (v / ((i as f64) + 1.0).sqrt()).cos())
        .product();
    sum_sq - prod_cos + 1.0
}

#[test]
fn ackley_4d_converges() {
    let fbest = run_seeded(vec![2.0; 4], 0.6, 50000, 1e-6, ackley);
    assert!(fbest < 1e-3, "ackley optimum not reached: {fbest}");
}

#[test]
fn griewank_5d_converges() {
    let fbest = run_seeded(vec![1.5; 5], 0.4, 60000, 1e-8, griewank);
    assert!(fbest < 5e-3, "griewank optimum not reached: {fbest}");
}

#[test]
fn schwefel_3d_converges() {
    // Start near the global minimizer at ~420.9687; use diagonal covariance to respect axes.
    let fbest = run_seeded_mode(
        vec![420.0; 3],
        30.0,
        120000,
        1e-6,
        2024,
        CovarianceModeKind::Diagonal,
        schwefel,
    );
    assert!(fbest < 5e-2, "schwefel optimum not reached: {fbest}");
}
