use fastcma::CovarianceModeKind;
use fastcma::test_utils::run_seeded_mode;

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

fn schwefel_2_26(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|v| v * v.abs().sqrt().sin()).sum();
    418.982_887_272_433_9 * n - sum
}

#[test]
fn very_hard_lite_converges() {
    let cases = [
        (
            "ackley-16d-lite",
            ackley as fn(&[f64]) -> f64,
            vec![1.2; 16],
            0.45,
            60_000,
            1e-6,
            2e-2,
            CovarianceModeKind::Full,
        ),
        (
            "rastrigin-16d-lite",
            rastrigin,
            vec![0.3; 16],
            0.45,
            80_000,
            1e-8,
            25.0,
            CovarianceModeKind::Full,
        ),
        (
            "schwefel-2.26-12d-lite",
            schwefel_2_26,
            vec![420.0; 12],
            35.0,
            90_000,
            1e-6,
            5.0,
            CovarianceModeKind::Diagonal,
        ),
    ];

    for (name, f, x0, sigma, maxfevals, ftarget, tol, mode) in cases {
        let fbest = run_seeded_mode(x0, sigma, maxfevals, ftarget, 7777, mode, f);
        assert!(fbest < tol, "{name} failed: fbest={fbest} tol={tol}");
    }
}
