use fastcma::test_utils::run_seeded_mode;
use fastcma::CovarianceModeKind;

fn zakharov(x: &[f64]) -> f64 {
    let sum1: f64 = x.iter().map(|v| v * v).sum();
    let sum2: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| 0.5 * ((i + 1) as f64) * v)
        .sum();
    sum1 + sum2 * sum2 + sum2 * sum2 * sum2 * sum2
}

fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|v| 1.0 + (*v - 1.0) / 4.0).collect();
    let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
    let term3 = (w.last().unwrap() - 1.0).powi(2) * (1.0 + (2.0 * std::f64::consts::PI * w.last().unwrap()).sin().powi(2));
    let mut sum = 0.0;
    for i in 0..w.len() - 1 {
        sum += (w[i] - 1.0).powi(2) * (1.0 + (std::f64::consts::PI * w[i + 1]).sin().powi(2));
    }
    term1 + sum + term3
}

fn dixon_price(x: &[f64]) -> f64 {
    let mut s = (x[0] - 1.0).powi(2);
    for i in 1..x.len() {
        let xi = x[i];
        let xprev = x[i - 1];
        s += ((2.0 * xi * xi) - xprev).powi(2);
    }
    s
}

fn powell(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for chunk in x.chunks_exact(4) {
        let x1 = chunk[0];
        let x2 = chunk[1];
        let x3 = chunk[2];
        let x4 = chunk[3];
        s += (x1 + 10.0 * x2).powi(2);
        s += 5.0 * (x3 - x4).powi(2);
        s += (x2 - 2.0 * x3).powi(4);
        s += 10.0 * (x1 - x4).powi(4);
    }
    s
}

fn styblinski_tang(x: &[f64]) -> f64 {
    x.iter().map(|v| v.powi(4) - 16.0 * v * v + 5.0 * v).sum::<f64>() / 2.0
}

fn bohachevsky(x: &[f64]) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    x0 * x0 + 2.0 * x1 * x1 - 0.3 * (3.0 * x0 * std::f64::consts::PI).cos()
        - 0.4 * (4.0 * x1 * std::f64::consts::PI).cos()
        + 0.7
}

fn bukin6(x: &[f64]) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    100.0 * (x1 - 0.01 * x0 * x0).abs().sqrt() + 0.01 * (x0 + 10.0).abs()
}

fn dropwave(x: &[f64]) -> f64 {
    let r2 = x[0] * x[0] + x[1] * x[1];
    let num = (r2.sqrt().cos()).abs() + 1.0;
    let den = 1.0 + 0.5 * r2;
    -(num / den)
}

fn alpine_n1(x: &[f64]) -> f64 {
    x.iter().map(|v| (v * v.sin()).abs() + 0.1 * v.abs()).sum()
}

fn elliptic(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    x.iter()
        .enumerate()
        .map(|(i, v)| 10f64.powf(6.0 * (i as f64) / (n - 1.0)) * v * v)
        .sum()
}

fn salomon(x: &[f64]) -> f64 {
    let r = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    1.0 - (2.0 * std::f64::consts::PI * r).cos() + 0.1 * r
}

fn quartic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| ((i as f64) + 1.0) * v.powi(4))
        .sum()
}

fn schwefel_1_2(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut acc = 0.0;
    for v in x {
        acc += *v;
        s += acc * acc;
    }
    s
}

fn schwefel_2_22(x: &[f64]) -> f64 {
    let sum_abs: f64 = x.iter().map(|v| v.abs()).sum();
    let prod_abs: f64 = x.iter().map(|v| v.abs()).product();
    sum_abs + prod_abs
}

fn bent_cigar(x: &[f64]) -> f64 {
    let mut s = x[0] * x[0];
    for v in &x[1..] {
        s += 1e6 * v * v;
    }
    s
}

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

fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / 4000.0;
    let prod_cos: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| (v / ((i as f64) + 1.0).sqrt()).cos())
        .product();
    sum_sq - prod_cos + 1.0
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

fn sum_squares(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| ((i as f64) + 1.0) * v * v)
        .sum()
}

fn discus(x: &[f64]) -> f64 {
    let mut s = 1e6 * x[0] * x[0];
    for v in &x[1..] {
        s += v * v;
    }
    s
}

fn different_powers(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    x.iter()
        .enumerate()
        .map(|(i, v)| v.abs().powf(2.0 + 10.0 * (i as f64) / (n - 1.0)))
        .sum()
}

fn katsuura(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut prod = 1.0;
    for (i, &xi) in x.iter().enumerate() {
        let mut sum = 0.0;
        for j in 1..=32 {
            let two = 2f64.powi(j);
            let val = two * xi;
            sum += (val - val.round()).abs() / two;
        }
        prod *= (1.0 + (i as f64 + 1.0) * sum).powf(10.0 / n.powf(1.2));
    }
    prod - 1.0
}

fn weierstrass(x: &[f64]) -> f64 {
    let a: f64 = 0.5;
    let b: f64 = 3.0;
    let kmax = 20;
    let mut sum1 = 0.0;
    for &xi in x {
        for k in 0..=kmax {
            let ak = a.powi(k);
            let bk = b.powi(k as i32);
            sum1 += ak * (2.0 * std::f64::consts::PI * bk * (xi + 0.5)).cos();
        }
    }
    let mut sum2 = 0.0;
    for k in 0..=kmax {
        let ak = a.powi(k);
        let bk = b.powi(k as i32);
        sum2 += ak * (2.0 * std::f64::consts::PI * bk * 0.5).cos();
    }
    sum1 - (x.len() as f64) * sum2
}

fn schwefel_2_26(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x
        .iter()
        .map(|&xi| xi * xi.abs().sqrt().sin())
        .sum();
    418.982_887_272_433_9 * n - sum
}

fn happy_cat(x: &[f64]) -> f64 {
    let norm2: f64 = x.iter().map(|v| v * v).sum();
    let sumx: f64 = x.iter().sum();
    let n = x.len() as f64;
    (norm2 - n).abs().sqrt().sqrt() + (0.5 * norm2 + sumx) / n + 0.5
}

fn hgbat(x: &[f64]) -> f64 {
    let norm2: f64 = x.iter().map(|v| v * v).sum();
    let sumx: f64 = x.iter().sum();
    let n = x.len() as f64;
    (norm2 * norm2 - sumx * sumx).abs().sqrt() + (0.5 * norm2 + sumx) / n + 0.5
}

fn expanded_schaffer6(x: &[f64]) -> f64 {
    fn f6(a: f64, b: f64) -> f64 {
        let r2 = a * a + b * b;
        0.5 + (r2.sqrt().sin().powi(2) - 0.5) / (1.0 + 0.001 * r2 * r2)
    }

    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += f6(x[i], x[i + 1]);
    }
    s += f6(x[x.len() - 1], x[0]);
    s
}

fn schaffer_f7(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let r2 = x[i] * x[i] + x[i + 1] * x[i + 1];
        let term = r2.sqrt().sqrt();
        acc += term * ((50.0 * r2.powf(0.1)).sin().powi(2) + 1.0);
    }
    acc / ((x.len() - 1) as f64)
}

fn sharp_ridge(x: &[f64]) -> f64 {
    let tail_sum: f64 = x[1..].iter().map(|v| v * v).sum();
    x[0] * x[0] + 100.0 * tail_sum.sqrt()
}

fn step(x: &[f64]) -> f64 {
    x.iter()
        .map(|v| (v + 0.5).floor())
        .map(|k| k * k)
        .sum()
}

struct Case<'a> {
    name: &'a str,
    f: fn(&[f64]) -> f64,
    x0: Vec<f64>,
    sigma: f64,
    maxfevals: usize,
    ftarget: f64,
    tol: f64,
    mode: CovarianceModeKind,
}

#[test]
fn hard_suite_converges() {
    let cases: &[Case] = &[
        Case { name: "zakharov-8d", f: zakharov, x0: vec![0.6; 8], sigma: 0.4, maxfevals: 80_000, ftarget: 1e-12, tol: 1e-6, mode: CovarianceModeKind::Full },
        Case { name: "levy-6d", f: levy, x0: vec![2.0; 6], sigma: 0.5, maxfevals: 120_000, ftarget: 1e-10, tol: 1e-4, mode: CovarianceModeKind::Full },
        Case { name: "dixon-price-8d", f: dixon_price, x0: vec![2.5; 8], sigma: 0.6, maxfevals: 180_000, ftarget: 1e-10, tol: 0.35, mode: CovarianceModeKind::Full },
        Case { name: "powell-4d", f: powell, x0: vec![3.0; 4], sigma: 0.8, maxfevals: 60_000, ftarget: 1e-12, tol: 1e-4, mode: CovarianceModeKind::Full },
        Case { name: "styblinski-tang-6d", f: styblinski_tang, x0: vec![1.5; 6], sigma: 0.6, maxfevals: 160_000, ftarget: -220.0, tol: 2.0, mode: CovarianceModeKind::Full },
        Case { name: "bohachevsky", f: bohachevsky, x0: vec![1.5, -1.2], sigma: 0.5, maxfevals: 40_000, ftarget: 1e-12, tol: 1e-4, mode: CovarianceModeKind::Full },
        Case { name: "bukin6", f: bukin6, x0: vec![-8.0, 3.0], sigma: 1.5, maxfevals: 80_000, ftarget: 0.0, tol: 0.05, mode: CovarianceModeKind::Full },
        Case { name: "dropwave", f: dropwave, x0: vec![1.2, -1.2], sigma: 0.8, maxfevals: 60_000, ftarget: -1.0, tol: 0.05, mode: CovarianceModeKind::Full },
        Case { name: "alpine-n1-8d", f: alpine_n1, x0: vec![1.0; 8], sigma: 0.5, maxfevals: 80_000, ftarget: 1e-12, tol: 1e-5, mode: CovarianceModeKind::Full },
        Case { name: "elliptic-10d", f: elliptic, x0: vec![0.3; 10], sigma: 0.35, maxfevals: 80_000, ftarget: 1e-16, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "salomon-8d", f: salomon, x0: vec![1.0; 8], sigma: 0.5, maxfevals: 140_000, ftarget: 1e-8, tol: 0.35, mode: CovarianceModeKind::Full },
        Case { name: "quartic-8d", f: quartic, x0: vec![0.7; 8], sigma: 0.4, maxfevals: 60_000, ftarget: 1e-12, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "schwefel-1.2-6d", f: schwefel_1_2, x0: vec![0.8; 6], sigma: 0.5, maxfevals: 80_000, ftarget: 1e-12, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "schwefel-2.22-6d", f: schwefel_2_22, x0: vec![0.5; 6], sigma: 0.5, maxfevals: 60_000, ftarget: 1e-12, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "bent-cigar-10d", f: bent_cigar, x0: vec![1.0; 10], sigma: 0.6, maxfevals: 120_000, ftarget: 1e-16, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "rastrigin-10d", f: rastrigin, x0: vec![0.3; 10], sigma: 0.4, maxfevals: 260_000, ftarget: 1e-10, tol: 15.0, mode: CovarianceModeKind::Full },
        Case { name: "ackley-10d", f: ackley, x0: vec![1.5; 10], sigma: 0.45, maxfevals: 180_000, ftarget: 1e-8, tol: 5e-3, mode: CovarianceModeKind::Full },
        Case { name: "griewank-10d", f: griewank, x0: vec![1.2; 10], sigma: 0.4, maxfevals: 140_000, ftarget: 1e-10, tol: 5e-3, mode: CovarianceModeKind::Full },
        Case { name: "rosenbrock-6d", f: rosenbrock, x0: vec![-1.2, 1.0, -0.5, 1.0, -1.0, 1.0], sigma: 0.6, maxfevals: 220_000, ftarget: 1e-10, tol: 1e-3, mode: CovarianceModeKind::Full },
        Case { name: "sum-squares-12d", f: sum_squares, x0: vec![0.4; 12], sigma: 0.35, maxfevals: 60_000, ftarget: 1e-12, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
    ];

    for c in cases {
        let fbest = run_seeded_mode(
            c.x0.clone(),
            c.sigma,
            c.maxfevals,
            c.ftarget,
            4242,
            c.mode,
            c.f,
        );
        assert!(
            fbest < c.tol,
            "{} failed: fbest={fbest} tol={} maxfevals={}",
            c.name,
            c.tol,
            c.maxfevals
        );
    }
}

#[test]
#[ignore]
fn very_hard_suite_converges() {
    let cases: &[Case] = &[
        Case { name: "discus-25d", f: discus, x0: vec![0.3; 25], sigma: 0.5, maxfevals: 120_000, ftarget: 1e-16, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "different-powers-20d", f: different_powers, x0: vec![0.4; 20], sigma: 0.4, maxfevals: 140_000, ftarget: 1e-16, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
        Case { name: "katsuura-16d", f: katsuura, x0: vec![0.3; 16], sigma: 0.5, maxfevals: 500_000, ftarget: 1e-12, tol: 0.5, mode: CovarianceModeKind::Diagonal },
        Case { name: "weierstrass-16d", f: weierstrass, x0: vec![0.2; 16], sigma: 0.45, maxfevals: 400_000, ftarget: 1e-12, tol: 5e-2, mode: CovarianceModeKind::Diagonal },
        Case { name: "schwefel-2.26-30d", f: schwefel_2_26, x0: vec![420.0; 30], sigma: 50.0, maxfevals: 400_000, ftarget: 1e-8, tol: 2.0, mode: CovarianceModeKind::Diagonal },
        Case { name: "happy-cat-16d", f: happy_cat, x0: vec![-2.0; 16], sigma: 0.6, maxfevals: 200_000, ftarget: 1e-12, tol: 0.1, mode: CovarianceModeKind::Diagonal },
        Case { name: "hgbat-16d", f: hgbat, x0: vec![-2.0; 16], sigma: 0.6, maxfevals: 200_000, ftarget: 1e-12, tol: 0.5, mode: CovarianceModeKind::Diagonal },
        Case { name: "expanded-schaffer6-16d", f: expanded_schaffer6, x0: vec![0.6; 16], sigma: 0.5, maxfevals: 200_000, ftarget: 1e-12, tol: 0.2, mode: CovarianceModeKind::Diagonal },
        Case { name: "schaffer-f7-12d", f: schaffer_f7, x0: vec![0.5; 12], sigma: 0.6, maxfevals: 220_000, ftarget: 1e-12, tol: 0.2, mode: CovarianceModeKind::Diagonal },
        Case { name: "sharp-ridge-20d", f: sharp_ridge, x0: vec![1.0; 20], sigma: 0.6, maxfevals: 200_000, ftarget: 1e-16, tol: 2.0, mode: CovarianceModeKind::Diagonal },
        Case { name: "step-30d", f: step, x0: vec![1.2; 30], sigma: 0.6, maxfevals: 80_000, ftarget: 1e-16, tol: 1e-3, mode: CovarianceModeKind::Diagonal },
        Case { name: "zakharov-24d", f: zakharov, x0: vec![0.7; 24], sigma: 0.35, maxfevals: 140_000, ftarget: 1e-12, tol: 1e-4, mode: CovarianceModeKind::Full },
        Case { name: "levy-14d", f: levy, x0: vec![2.0; 14], sigma: 0.5, maxfevals: 160_000, ftarget: 1e-10, tol: 1e-4, mode: CovarianceModeKind::Full },
        Case { name: "powell-12d", f: powell, x0: vec![3.0; 12], sigma: 0.8, maxfevals: 140_000, ftarget: 1e-12, tol: 1e-3, mode: CovarianceModeKind::Full },
        Case { name: "styblinski-tang-10d", f: styblinski_tang, x0: vec![1.5; 10], sigma: 0.6, maxfevals: 200_000, ftarget: -400.0, tol: 5.0, mode: CovarianceModeKind::Full },
        Case { name: "ackley-30d", f: ackley, x0: vec![1.5; 30], sigma: 0.5, maxfevals: 300_000, ftarget: 1e-6, tol: 1e-2, mode: CovarianceModeKind::Full },
        Case { name: "rastrigin-30d", f: rastrigin, x0: vec![0.2; 30], sigma: 0.5, maxfevals: 400_000, ftarget: 1e-6, tol: 120.0, mode: CovarianceModeKind::Full },
        Case { name: "rosenbrock-20d", f: rosenbrock, x0: vec![-1.2, 1.0].into_iter().cycle().take(20).collect(), sigma: 0.6, maxfevals: 500_000, ftarget: 1e-8, tol: 1e-2, mode: CovarianceModeKind::Full },
        Case { name: "griewank-30d", f: griewank, x0: vec![1.2; 30], sigma: 0.4, maxfevals: 260_000, ftarget: 1e-10, tol: 1e-2, mode: CovarianceModeKind::Full },
        Case { name: "elliptic-30d", f: elliptic, x0: vec![0.3; 30], sigma: 0.35, maxfevals: 120_000, ftarget: 1e-16, tol: 1e-6, mode: CovarianceModeKind::Diagonal },
    ];

    for c in cases {
        let fbest = run_seeded_mode(
            c.x0.clone(),
            c.sigma,
            c.maxfevals,
            c.ftarget,
            1337,
            c.mode,
            c.f,
        );
        assert!(
            fbest < c.tol,
            "{} failed: fbest={fbest} tol={} maxfevals={}",
            c.name,
            c.tol,
            c.maxfevals
        );
    }
}
