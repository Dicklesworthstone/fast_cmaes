#![feature(portable_simd)]

use core::simd::{Simd, num::SimdFloat};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64;

pub mod ffi;

use nalgebra::DMatrix;

#[cfg(all(feature = "python", feature = "numpy_support"))]
use numpy::PyReadonlyArray1;

const SIMD_LANES: usize = 4;
type SimdF64 = Simd<f64, SIMD_LANES>;

fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = SimdF64::splat(0.0);
    let mut i = 0;
    while i + SIMD_LANES <= len {
        let va = SimdF64::from_slice(&a[i..i + SIMD_LANES]);
        let vb = SimdF64::from_slice(&b[i..i + SIMD_LANES]);
        sum += va * vb;
        i += SIMD_LANES;
    }
    let mut scalar = sum.reduce_sum();
    while i < len {
        scalar += a[i] * b[i];
        i += 1;
    }
    scalar
}

fn square_sum_simd(x: &[f64]) -> f64 {
    dot_simd(x, x)
}

#[cfg(feature = "eigen_lapack")]
fn symmetric_eigen_from_data(data: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mat = DMatrix::from_row_slice(n, n, data);
    let se = nalgebra_lapack::SymmetricEigen::new(mat);
    let eigenvalues: Vec<f64> = se.eigenvalues.iter().copied().collect();
    let mut eigenbasis = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            eigenbasis[i * n + j] = se.eigenvectors[(i, j)];
        }
    }
    (eigenvalues, eigenbasis)
}

#[cfg(not(feature = "eigen_lapack"))]
fn symmetric_eigen_from_data(data: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mat = DMatrix::from_row_slice(n, n, data);
    let se = nalgebra::linalg::SymmetricEigen::new(mat);
    let eigenvalues: Vec<f64> = se.eigenvalues.iter().copied().collect();
    let mut eigenbasis = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            eigenbasis[i * n + j] = se.eigenvectors[(i, j)];
        }
    }
    (eigenvalues, eigenbasis)
}

#[derive(Clone, Copy)]
pub enum CovarianceModeKind {
    Full,
    Diagonal,
}

struct CmaesParameters {
    #[allow(dead_code)]
    dimension: usize,
    lam: usize,
    mu: usize,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    #[allow(dead_code)]
    chi_n: f64,
    lazy_gap_evals: f64,
}

impl CmaesParameters {
    fn new(n: usize, popsize: Option<usize>) -> Self {
        let n_f = n as f64;
        let lam = popsize.unwrap_or_else(|| {
            let v = 4.0 + 3.0 * n_f.ln();
            if v < 1.0 { 1 } else { v.floor() as usize }
        });
        let mu = lam / 2;
        let mut weights = vec![0.0; lam];
        for (i, weight) in weights.iter_mut().enumerate() {
            if i < mu {
                let num = (lam as f64) / 2.0 + 0.5;
                *weight = num.ln() - ((i + 1) as f64).ln();
            } else {
                *weight = 0.0;
            }
        }
        let w_sum: f64 = weights[..mu].iter().sum();
        if w_sum != 0.0 {
            for w in &mut weights {
                *w /= w_sum;
            }
        }
        let mueff_num: f64 = weights[..mu].iter().sum::<f64>().powi(2);
        let mueff_den: f64 = weights[..mu].iter().map(|w| w * w).sum();
        let mueff = if mueff_den > 0.0 {
            mueff_num / mueff_den
        } else {
            1.0
        };
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));
        let cc = (4.0 + mueff / n_f) / (n_f + 4.0 + 2.0 * mueff / n_f);
        let cs = (mueff + 2.0) / (n_f + mueff + 5.0);
        let c1 = 2.0 / ((n_f + 1.3).powi(2) + mueff);
        let cmu = {
            let up = 2.0 * (mueff - 2.0 + 1.0 / mueff);
            let down = (n_f + 2.0).powi(2) + mueff;
            let v = up / down;
            v.min(1.0 - c1)
        };
        let damps = 2.0 * mueff / (lam as f64) + 0.3 + cs;
        let lazy_gap_evals = 0.5 * n_f * (lam as f64) * 1.0 / (c1 + cmu) / (n_f * n_f);
        Self {
            dimension: n,
            lam,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            lazy_gap_evals,
        }
    }
}

struct BestSolution {
    x: Vec<f64>,
    f: f64,
    evals: usize,
    initialized: bool,
}

impl BestSolution {
    fn new(dim: usize) -> Self {
        Self {
            x: vec![0.0; dim],
            f: f64::INFINITY,
            evals: 0,
            initialized: false,
        }
    }
    fn update(&mut self, x: &[f64], f: f64, evals: usize) {
        if !self.initialized || f < self.f {
            self.x = x.to_vec();
            self.f = f;
            self.evals = evals;
            self.initialized = true;
        }
    }
}

struct FullCovariance {
    n: usize,
    data: Vec<f64>,
    eigenbasis: Vec<f64>,
    eigenvalues: Vec<f64>,
    invsqrt: Vec<f64>,
    condition_number: f64,
    updated_eval: usize,
}

impl FullCovariance {
    fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let eigenbasis = data.clone();
        let eigenvalues = vec![1.0; n];
        let invsqrt = data.clone();
        Self {
            n,
            data,
            eigenbasis,
            eigenvalues,
            invsqrt,
            condition_number: 1.0,
            updated_eval: 0,
        }
    }
    fn diag(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.data[i * self.n + i]).collect()
    }
    fn multiply_with(&mut self, factor: f64) {
        self.data.par_iter_mut().for_each(|v| *v *= factor);
    }
    fn addouter(&mut self, b: &[f64], factor: f64) {
        let n = self.n;
        let b_vec = b.to_vec();
        self.data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                let bi = b_vec[i];
                let scale = factor * bi;
                for j in 0..n {
                    row[j] += scale * b_vec[j];
                }
            });
    }
    fn enforce_symmetry(&mut self) {
        let n = self.n;
        let data_ptr: *mut f64 = self.data.as_mut_ptr();
        (0..n).for_each(|i| {
            for j in 0..i {
                let idx1 = i * n + j;
                let idx2 = j * n + i;
                unsafe {
                    let v1 = *data_ptr.add(idx1);
                    let v2 = *data_ptr.add(idx2);
                    let avg = 0.5 * (v1 + v2);
                    *data_ptr.add(idx1) = avg;
                    *data_ptr.add(idx2) = avg;
                }
            }
        });
    }
    fn update_eigensystem(&mut self, current_eval: usize, lazy_gap_evals: f64) {
        if (current_eval as f64) <= (self.updated_eval as f64) + lazy_gap_evals {
            return;
        }
        self.enforce_symmetry();
        let (mut eigs, basis) = symmetric_eigen_from_data(&self.data, self.n);
        let mut min_ev = f64::INFINITY;
        let mut max_ev = 0.0;
        for &ev in &eigs {
            if ev < min_ev {
                min_ev = ev;
            }
            if ev > max_ev {
                max_ev = ev;
            }
        }
        if min_ev <= 0.0 {
            // Floor tiny/negative eigenvalues to keep the eigensystem usable without panicking.
            let eps = 1e-20;
            for ev in &mut eigs {
                if *ev < eps {
                    *ev = eps;
                }
            }
            min_ev = eigs.iter().fold(f64::INFINITY, |m, &v| m.min(v));
            max_ev = eigs.iter().fold(0.0f64, |m, &v| m.max(v));
        }
        let n = self.n;
        let mut invsqrt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..n {
                    let bik = basis[i * n + k];
                    let bjk = basis[j * n + k];
                    sum += bik * bjk / eigs[k].sqrt();
                }
                invsqrt[i * n + j] = sum;
                invsqrt[j * n + i] = sum;
            }
        }
        self.eigenbasis = basis;
        self.eigenvalues = eigs;
        self.invsqrt = invsqrt;
        self.condition_number = max_ev / min_ev;
        self.updated_eval = current_eval;
    }
    fn mahalanobis_norm(&self, dx: &[f64], tmp: &mut [f64]) -> f64 {
        let n = self.n;
        for (i, tmp_val) in tmp.iter_mut().enumerate().take(n) {
            let row_start = i * n;
            let mut sum = 0.0;
            for (j, &dx_val) in dx.iter().enumerate().take(n) {
                sum += self.invsqrt[row_start + j] * dx_val;
            }
            *tmp_val = sum;
        }
        square_sum_simd(tmp).sqrt()
    }
    fn invsqrt_mul(&self, y: &[f64], out: &mut [f64]) {
        let n = self.n;
        for (i, out_val) in out.iter_mut().enumerate().take(n) {
            let row_start = i * n;
            let mut sum = 0.0;
            for (j, &y_val) in y.iter().enumerate().take(n) {
                sum += self.invsqrt[row_start + j] * y_val;
            }
            *out_val = sum;
        }
    }
    fn mul_eigenbasis_vec(&self, z: &[f64], out: &mut [f64]) {
        let n = self.n;
        for (i, out_val) in out.iter_mut().enumerate().take(n) {
            let row_start = i * n;
            let mut sum = 0.0;
            for (j, &z_val) in z.iter().enumerate().take(n) {
                sum += self.eigenbasis[row_start + j] * z_val;
            }
            *out_val = sum;
        }
    }
}

struct DiagonalCovariance {
    n: usize,
    diag: Vec<f64>,
    invsqrt_diag: Vec<f64>,
    condition_number: f64,
}

impl DiagonalCovariance {
    fn identity(n: usize) -> Self {
        Self {
            n,
            diag: vec![1.0; n],
            invsqrt_diag: vec![1.0; n],
            condition_number: 1.0,
        }
    }
    fn update_internals(&mut self) {
        let mut min_ev = f64::INFINITY;
        let mut max_ev = 0.0;
        for &d in &self.diag {
            let val = if d <= 0.0 { 1e-30 } else { d };
            if val < min_ev {
                min_ev = val;
            }
            if val > max_ev {
                max_ev = val;
            }
        }
        for (inv, &d) in self.invsqrt_diag.iter_mut().zip(self.diag.iter()) {
            let val = if d <= 0.0 { 1e-30 } else { d };
            *inv = 1.0 / val.sqrt();
        }
        self.condition_number = max_ev / min_ev;
    }
}

enum CovarianceMode {
    Full(FullCovariance),
    Diagonal(DiagonalCovariance),
}

impl CovarianceMode {
    #[allow(dead_code)]
    fn dimension(&self) -> usize {
        match self {
            CovarianceMode::Full(c) => c.n,
            CovarianceMode::Diagonal(c) => c.n,
        }
    }
    fn diag(&self) -> Vec<f64> {
        match self {
            CovarianceMode::Full(c) => c.diag(),
            CovarianceMode::Diagonal(c) => c.diag.clone(),
        }
    }
    fn update_eigensystem(&mut self, current_eval: usize, lazy_gap_evals: f64) {
        match self {
            CovarianceMode::Full(c) => c.update_eigensystem(current_eval, lazy_gap_evals),
            CovarianceMode::Diagonal(c) => {
                if current_eval > 0 {
                    c.update_internals();
                }
            }
        }
    }
    fn multiply_with(&mut self, factor: f64) {
        match self {
            CovarianceMode::Full(c) => c.multiply_with(factor),
            CovarianceMode::Diagonal(c) => {
                for d in &mut c.diag {
                    *d *= factor;
                }
                c.update_internals();
            }
        }
    }
    fn addouter(&mut self, b: &[f64], factor: f64) {
        match self {
            CovarianceMode::Full(c) => c.addouter(b, factor),
            CovarianceMode::Diagonal(c) => {
                for i in 0..c.n {
                    c.diag[i] += factor * b[i] * b[i];
                }
                c.update_internals();
            }
        }
    }
    fn mahalanobis_norm(&self, dx: &[f64], tmp: &mut [f64]) -> f64 {
        match self {
            CovarianceMode::Full(c) => c.mahalanobis_norm(dx, tmp),
            CovarianceMode::Diagonal(c) => {
                for i in 0..c.n {
                    let val = if c.diag[i] <= 0.0 { 1e-30 } else { c.diag[i] };
                    tmp[i] = dx[i] / val.sqrt();
                }
                square_sum_simd(&tmp[..c.n]).sqrt()
            }
        }
    }
    fn invsqrt_mul(&self, y: &[f64], out: &mut [f64]) {
        match self {
            CovarianceMode::Full(c) => c.invsqrt_mul(y, out),
            CovarianceMode::Diagonal(c) => {
                for i in 0..c.n {
                    out[i] = y[i] * c.invsqrt_diag[i];
                }
            }
        }
    }
    fn condition_number(&self) -> f64 {
        match self {
            CovarianceMode::Full(c) => c.condition_number,
            CovarianceMode::Diagonal(c) => c.condition_number,
        }
    }
    fn max_eigenvalue(&self) -> f64 {
        match self {
            CovarianceMode::Full(c) => c.eigenvalues.iter().cloned().fold(0.0f64, f64::max),
            CovarianceMode::Diagonal(c) => c.diag.iter().cloned().fold(0.0f64, f64::max),
        }
    }
    fn sample<R: Rng + ?Sized>(
        &self,
        sigma: f64,
        rng: &mut R,
        z_buf: &mut [f64],
        y_buf: &mut [f64],
    ) {
        match self {
            CovarianceMode::Full(c) => {
                let n = c.n;
                for i in 0..n {
                    let ev = if c.eigenvalues.is_empty() {
                        1.0
                    } else {
                        c.eigenvalues[i].max(0.0)
                    };
                    let s: f64 = rng.sample(StandardNormal);
                    z_buf[i] = sigma * ev.sqrt() * s;
                }
                c.mul_eigenbasis_vec(&z_buf[..n], &mut y_buf[..n]);
            }
            CovarianceMode::Diagonal(c) => {
                let n = c.n;
                for i in 0..n {
                    let var = if c.diag[i] <= 0.0 { 1e-30 } else { c.diag[i] };
                    let s: f64 = rng.sample(StandardNormal);
                    y_buf[i] = sigma * var.sqrt() * s;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum TerminationReason {
    MaxFevals(usize),
    FTarget(f64),
    Condition(f64),
    TolFun(f64),
    TolX(f64),
}

pub struct CmaesState {
    n: usize,
    params: CmaesParameters,
    maxfevals: usize,
    ftarget: Option<f64>,
    sigma0: f64,
    noise_cfg: Option<NoiseConfig>,
    noise_cooldown: usize,
    rng: StdRng,
    xmean: Vec<f64>,
    sigma: f64,
    pc: Vec<f64>,
    ps: Vec<f64>,
    cov: CovarianceMode,
    counteval: usize,
    fitvals: Vec<f64>,
    best: BestSolution,
    z_buf: Vec<f64>,
    y_buf: Vec<f64>,
    tmp_buf: Vec<f64>,
    pending_arx: Vec<Vec<f64>>,
    pending_fitvals: Vec<f64>,
}

#[derive(Clone, Copy)]
struct NoiseConfig {
    threshold_rel: f64,
    sigma_expand: f64,
    max_sigma_factor: f64,
    cooldown_iters: usize,
}

impl CmaesState {
    pub fn new(
        xstart: Vec<f64>,
        sigma: f64,
        popsize: Option<usize>,
        ftarget: Option<f64>,
        maxfevals: Option<usize>,
        covariance_mode: CovarianceModeKind,
    ) -> Self {
        let n = xstart.len();
        let params = CmaesParameters::new(n, popsize);
        let maxfevals_val = maxfevals.unwrap_or_else(|| {
            let n_f = n as f64;
            let lam_f = params.lam as f64;
            let v = 100.0 * lam_f + 150.0 * (n_f + 3.0).powi(2) * lam_f.sqrt();
            if v < 1.0 { 1 } else { v.round() as usize }
        });
        let cov = match covariance_mode {
            CovarianceModeKind::Full => CovarianceMode::Full(FullCovariance::identity(n)),
            CovarianceModeKind::Diagonal => {
                CovarianceMode::Diagonal(DiagonalCovariance::identity(n))
            }
        };
        let best = BestSolution::new(n);
        Self {
            n,
            params,
            maxfevals: maxfevals_val,
            ftarget,
            sigma0: sigma,
            noise_cfg: None,
            noise_cooldown: 0,
            rng: StdRng::from_entropy(),
            xmean: xstart,
            sigma,
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            cov,
            counteval: 0,
            fitvals: Vec::new(),
            best,
            z_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            tmp_buf: vec![0.0; n],
            pending_arx: Vec::new(),
            pending_fitvals: Vec::new(),
        }
    }

    pub fn new_with_seed(
        xstart: Vec<f64>,
        sigma: f64,
        popsize: Option<usize>,
        ftarget: Option<f64>,
        maxfevals: Option<usize>,
        covariance_mode: CovarianceModeKind,
        seed: u64,
    ) -> Self {
        let mut state = Self::new(xstart, sigma, popsize, ftarget, maxfevals, covariance_mode);
        state.rng = StdRng::seed_from_u64(seed);
        state
    }

    fn with_noise(mut self, cfg: NoiseConfig) -> Self {
        self.noise_cfg = Some(cfg);
        self
    }
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        self.cov
            .update_eigensystem(self.counteval, self.params.lazy_gap_evals);
        let lam = self.params.lam;
        let n = self.n;
        let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(lam);
        for _ in 0..lam {
            self.cov
                .sample(self.sigma, &mut self.rng, &mut self.z_buf, &mut self.y_buf);
            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                x.push(self.xmean[i] + self.y_buf[i]);
            }
            candidates.push(x);
        }
        candidates
    }
    fn ask_one(&mut self) -> Vec<f64> {
        self.cov
            .update_eigensystem(self.counteval, self.params.lazy_gap_evals);
        let n = self.n;
        self.cov
            .sample(self.sigma, &mut self.rng, &mut self.z_buf, &mut self.y_buf);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            x.push(self.xmean[i] + self.y_buf[i]);
        }
        x
    }
    pub fn tell(&mut self, arx: Vec<Vec<f64>>, fitvals: Vec<f64>) {
        let lam = self.params.lam;
        if arx.len() != lam || fitvals.len() != lam {
            eprintln!(
                "tell: expected {} candidates, got arx={}, fitvals={}",
                lam,
                arx.len(),
                fitvals.len()
            );
            return;
        }
        self.counteval += fitvals.len();
        let n = self.n;
        let params = &self.params;
        let xold = self.xmean.clone();

        let mut idx: Vec<usize> = (0..fitvals.len()).collect();
        idx.sort_by(|&i, &j| {
            let ai = fitvals[i];
            let aj = fitvals[j];
            match (ai.partial_cmp(&aj), ai.is_nan(), aj.is_nan()) {
                (Some(ord), false, false) => ord,
                (_, true, false) => Ordering::Greater,
                (_, false, true) => Ordering::Less,
                _ => Ordering::Equal,
            }
        });
        let mut arx_sorted = Vec::with_capacity(arx.len());
        let mut fit_sorted = Vec::with_capacity(fitvals.len());
        for i in idx {
            arx_sorted.push(arx[i].clone());
            fit_sorted.push(fitvals[i]);
        }
        self.fitvals = fit_sorted.clone();
        self.best
            .update(&arx_sorted[0], self.fitvals[0], self.counteval);

        let mu = params.mu;
        let weights = &params.weights;
        let mut new_xmean = vec![0.0; n];
        for k in 0..mu {
            let wk = weights[k];
            let xk = &arx_sorted[k];
            for i in 0..n {
                new_xmean[i] += wk * xk[i];
            }
        }
        self.xmean = new_xmean;

        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = self.xmean[i] - xold[i];
        }

        self.cov.invsqrt_mul(&y, &mut self.tmp_buf[..n]);
        let z = &self.tmp_buf[..n];

        let csn = (params.cs * (2.0 - params.cs) * params.mueff).sqrt() / self.sigma;
        for i in 0..n {
            self.ps[i] = (1.0 - params.cs) * self.ps[i] + csn * z[i];
        }

        let ccn = (params.cc * (2.0 - params.cc) * params.mueff).sqrt() / self.sigma;
        let sum_ps_sq = square_sum_simd(&self.ps[..n]);
        let n_f = n as f64;
        let factor =
            1.0 - (1.0 - params.cs).powf(2.0 * (self.counteval as f64) / (params.lam as f64));
        let hsig = if factor <= 0.0 {
            0.0
        } else {
            let val = sum_ps_sq / n_f / factor;
            if val < 2.0 + 4.0 / (n_f + 1.0) {
                1.0
            } else {
                0.0
            }
        };
        for i in 0..n {
            self.pc[i] = (1.0 - params.cc) * self.pc[i] + ccn * hsig * y[i];
        }

        let hsig_sq = hsig * hsig;
        let c1a = params.c1 * (1.0 - (1.0 - hsig_sq) * params.cc * (2.0 - params.cc));
        let w_sum: f64 = params.weights.iter().sum();
        self.cov.multiply_with(1.0 - c1a - params.cmu * w_sum);
        self.cov.addouter(&self.pc, params.c1);

        let n_f64 = n as f64;
        for (k, wk0) in params.weights.iter().enumerate() {
            let mut wk = *wk0;
            let mut dx = vec![0.0; n];
            for i in 0..n {
                dx[i] = arx_sorted[k][i] - xold[i];
            }
            if wk < 0.0 {
                let norm = self.cov.mahalanobis_norm(&dx, &mut self.tmp_buf[..n]);
                if norm > 0.0 {
                    let scale = n_f64 * (self.sigma / norm).powi(2);
                    wk *= scale;
                } else {
                    wk = 0.0;
                }
            }
            let scale = wk * params.cmu / (self.sigma * self.sigma);
            if scale != 0.0 {
                self.cov.addouter(&dx, scale);
            }
        }

        let cn = params.cs / params.damps;
        let exponent = cn * (sum_ps_sq / n_f - 1.0) / 2.0;
        let exponent_clamped = exponent.clamp(-1.0, 1.0);
        self.sigma *= exponent_clamped.exp();

        // Noise handling: if fitness spread is very small relative to best value,
        // we may be stuck under noisy evaluations. Expand sigma to escape.
        if let Some(cfg) = self.noise_cfg {
            if self.noise_cooldown > 0 {
                self.noise_cooldown -= 1;
            } else if self.fitvals.len() >= 2 {
                let best = self.fitvals[0];
                let mid = self.fitvals[self.fitvals.len() / 2];
                let scale = best.abs().max(1.0);
                let spread = (mid - best).abs() / scale;
                if spread < cfg.threshold_rel {
                    let max_sigma = self.sigma0 * cfg.max_sigma_factor;
                    self.sigma = (self.sigma * cfg.sigma_expand).min(max_sigma);
                    self.noise_cooldown = cfg.cooldown_iters;
                }
            }
        }
    }
    fn tell_one(&mut self, x: Vec<f64>, f: f64) {
        self.pending_arx.push(x);
        self.pending_fitvals.push(f);
        if self.pending_fitvals.len() == self.params.lam {
            let arx = std::mem::take(&mut self.pending_arx);
            let fit = std::mem::take(&mut self.pending_fitvals);
            self.tell(arx, fit);
        }
    }
    fn termination_reasons(&self) -> Vec<TerminationReason> {
        let mut res = Vec::new();
        if self.counteval == 0 {
            return res;
        }
        if self.counteval >= self.maxfevals {
            res.push(TerminationReason::MaxFevals(self.maxfevals));
        }
        if let Some(ft) = self.ftarget {
            if !self.fitvals.is_empty() && self.fitvals[0] <= ft {
                res.push(TerminationReason::FTarget(ft));
            }
        }
        if self.cov.condition_number() > 1e14 {
            res.push(TerminationReason::Condition(self.cov.condition_number()));
        }
        if self.fitvals.len() > 1 {
            let diff = self.fitvals[self.fitvals.len() - 1] - self.fitvals[0];
            if diff < 1e-12 {
                res.push(TerminationReason::TolFun(1e-12));
            }
        }
        let max_ev = self.cov.max_eigenvalue();
        if self.sigma * max_ev.sqrt() < 1e-11 {
            res.push(TerminationReason::TolX(1e-11));
        }
        res
    }
    pub fn has_terminated(&self) -> bool {
        !self.termination_reasons().is_empty()
    }
    pub fn result(&self) -> (Vec<f64>, f64, usize, usize, usize, Vec<f64>, Vec<f64>) {
        let xbest = if self.best.initialized {
            self.best.x.clone()
        } else {
            self.xmean.clone()
        };
        let fbest = if self.best.initialized {
            self.best.f
        } else {
            f64::INFINITY
        };
        let evals_best = if self.best.initialized {
            self.best.evals
        } else {
            0
        };
        let iterations = self.counteval / self.params.lam;
        let diag = self.cov.diag();
        let stds: Vec<f64> = diag.into_iter().map(|v| self.sigma * v.sqrt()).collect();
        (
            xbest,
            fbest,
            evals_best,
            self.counteval,
            iterations,
            self.xmean.clone(),
            stds,
        )
    }
    fn disp(&self, verb_modulo: usize) {
        if verb_modulo == 0 {
            return;
        }
        let lam_f = self.params.lam as f64;
        if lam_f == 0.0 {
            return;
        }
        let iteration = self.counteval as f64 / lam_f;
        if (iteration - 1.0).abs() < f64::EPSILON || (iteration % (10.0 * verb_modulo as f64)) < 1.0
        {
            println!("evals: ax-ratio max(std)   f-value");
        }
        if iteration <= 2.0 || (iteration % verb_modulo as f64) < 1.0 {
            let ax_ratio = self.cov.condition_number().sqrt();
            let max_diag = self.cov.diag().into_iter().fold(0.0f64, f64::max);
            let max_std = self.sigma * max_diag.sqrt();
            let f0 = if !self.fitvals.is_empty() {
                self.fitvals[0]
            } else {
                f64::NAN
            };
            println!(
                "{:5}: {:7.1} {:10.1e}  {}",
                self.counteval, ax_ratio, max_std, f0
            );
        }
    }
}

#[cfg(feature = "python")]
fn parse_covariance_mode(mode: Option<&str>) -> PyResult<CovarianceModeKind> {
    match mode {
        None => Ok(CovarianceModeKind::Full),
        Some(m) => {
            let ml = m.to_ascii_lowercase();
            match ml.as_str() {
                "full" => Ok(CovarianceModeKind::Full),
                "diag" | "diagonal" => Ok(CovarianceModeKind::Diagonal),
                other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown covariance_mode '{}', expected 'full' or 'diag'",
                    other
                ))),
            }
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "CMAES")]
struct PyCmaes {
    inner: CmaesState,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCmaes {
    #[new]
    #[pyo3(
        signature = (xstart, sigma, popsize=None, ftarget=None, maxfevals=None, covariance_mode=None)
    )]
    fn new(
        xstart: Vec<f64>,
        sigma: f64,
        popsize: Option<usize>,
        ftarget: Option<f64>,
        maxfevals: Option<usize>,
        covariance_mode: Option<&str>,
    ) -> PyResult<Self> {
        let mode = parse_covariance_mode(covariance_mode)?;
        let inner = CmaesState::new(xstart, sigma, popsize, ftarget, maxfevals, mode);
        Ok(Self { inner })
    }
    fn ask(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<f64>>> {
        let candidates = py.allow_threads(|| self.inner.ask());
        Ok(candidates)
    }
    fn ask_one(&mut self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let x = py.allow_threads(|| self.inner.ask_one());
        Ok(x)
    }
    fn tell(&mut self, py: Python<'_>, arx: Vec<Vec<f64>>, fitvals: Vec<f64>) {
        py.allow_threads(|| self.inner.tell(arx, fitvals));
    }
    fn tell_one(&mut self, py: Python<'_>, x: Vec<f64>, f: f64) {
        py.allow_threads(|| self.inner.tell_one(x, f));
    }
    fn stop<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new_bound(py);
        for r in self.inner.termination_reasons() {
            match r {
                TerminationReason::MaxFevals(v) => {
                    let _ = d.set_item("maxfevals", v);
                }
                TerminationReason::FTarget(v) => {
                    let _ = d.set_item("ftarget", v);
                }
                TerminationReason::Condition(v) => {
                    let _ = d.set_item("condition", v);
                }
                TerminationReason::TolFun(v) => {
                    let _ = d.set_item("tolfun", v);
                }
                TerminationReason::TolX(v) => {
                    let _ = d.set_item("tolx", v);
                }
            }
        }
        d
    }
    #[getter]
    fn result(&self) -> (Vec<f64>, f64, usize, usize, usize, Vec<f64>, Vec<f64>) {
        self.inner.result()
    }
    #[pyo3(signature = (verb_modulo=None))]
    fn disp(&self, py: Python<'_>, verb_modulo: Option<usize>) {
        let v = verb_modulo.unwrap_or(1);
        py.allow_threads(|| self.inner.disp(v));
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ff")]
struct PyFF;

#[cfg(feature = "python")]
#[pymethods]
impl PyFF {
    #[staticmethod]
    fn elli(x: Vec<f64>) -> f64 {
        let n = x.len();
        if n == 0 {
            return 0.0;
        }
        let aratio = 1e3_f64;
        let n_m1 = (n - 1) as f64;
        let mut s = 0.0;
        for (i, xi) in x.iter().enumerate() {
            let exp_arg = 2.0 * (i as f64) / n_m1;
            let w = aratio.powf(exp_arg);
            s += xi * xi * w;
        }
        s
    }
    #[staticmethod]
    fn sphere(x: Vec<f64>) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
    #[staticmethod]
    fn tablet(x: Vec<f64>) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let base: f64 = x.iter().map(|v| v * v).sum();
        base + (1e6_f64 - 1.0) * x[0] * x[0]
    }
    #[staticmethod]
    fn rosenbrock(x: Vec<f64>) -> PyResult<f64> {
        let n = x.len();
        if n < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dimension must be greater one",
            ));
        }
        let mut s = 0.0;
        for i in 0..(n - 1) {
            let xi = x[i];
            let xnext = x[i + 1];
            let term1 = 100.0 * (xi * xi - xnext) * (xi * xi - xnext);
            let term2 = (xi - 1.0) * (xi - 1.0);
            s += term1 + term2;
        }
        Ok(s)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(
    signature = (objective_fct, xstart, sigma, maxfevals=None, ftarget=None, verb_disp=None, covariance_mode=None, noise=false)
)]
fn fmin(
    py: Python<'_>,
    objective_fct: PyObject,
    xstart: Vec<f64>,
    sigma: f64,
    maxfevals: Option<usize>,
    ftarget: Option<f64>,
    verb_disp: Option<usize>,
    covariance_mode: Option<&str>,
    noise: bool,
) -> PyResult<(Vec<f64>, Py<PyCmaes>)> {
    let mode = parse_covariance_mode(covariance_mode)?;
    let mut es = CmaesState::new(xstart, sigma, None, ftarget, maxfevals, mode);
    if noise {
        es = es.with_noise(NoiseConfig {
            threshold_rel: 1e-3,
            sigma_expand: 1.6,
            max_sigma_factor: 5.0,
            cooldown_iters: 5,
        });
    }
    let disp_mod = verb_disp.unwrap_or(100);
    loop {
        if es.has_terminated() {
            break;
        }
        let x_candidates = py.allow_threads(|| es.ask());
        let mut fitvals: Vec<f64> = Vec::with_capacity(x_candidates.len());
        for x in &x_candidates {
            let x_py = PyList::new_bound(py, x);
            let val = objective_fct.call1(py, (x_py,))?;
            let f: f64 = val.extract(py)?;
            fitvals.push(f);
        }
        py.allow_threads(|| es.tell(x_candidates, fitvals));
        if disp_mod > 0 {
            py.allow_threads(|| es.disp(disp_mod));
        }
    }
    let (xbest, fbest, _evals_best, _counteval, _iters, xmean, _stds) = es.result();
    let xmean_val = {
        let x_py = PyList::new_bound(py, &xmean);
        let val = objective_fct.call1(py, (x_py,))?;
        let f: f64 = val.extract(py)?;
        f
    };
    let xmin = if fbest < xmean_val {
        xbest.clone()
    } else {
        xmean.clone()
    };
    let es_py = Py::new(py, PyCmaes { inner: es })?;
    Ok((xmin, es_py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(
    signature = (objective_fct, xstart, sigma, maxfevals=None, ftarget=None, verb_disp=None, covariance_mode=None, noise=false)
)]
fn fmin_vec(
    py: Python<'_>,
    objective_fct: PyObject,
    xstart: Vec<f64>,
    sigma: f64,
    maxfevals: Option<usize>,
    ftarget: Option<f64>,
    verb_disp: Option<usize>,
    covariance_mode: Option<&str>,
    noise: bool,
) -> PyResult<(Vec<f64>, Py<PyCmaes>)> {
    let mode = parse_covariance_mode(covariance_mode)?;
    let mut es = CmaesState::new(xstart, sigma, None, ftarget, maxfevals, mode);
    if noise {
        es = es.with_noise(NoiseConfig {
            threshold_rel: 1e-3,
            sigma_expand: 1.6,
            max_sigma_factor: 5.0,
            cooldown_iters: 5,
        });
    }
    let disp_mod = verb_disp.unwrap_or(100);
    loop {
        if es.has_terminated() {
            break;
        }
        let x_candidates = py.allow_threads(|| es.ask());
        let x_py = {
            let list = PyList::empty_bound(py);
            for x in &x_candidates {
                let row = PyList::new_bound(py, x);
                list.append(row)?;
            }
            list
        };
        let fit_py = objective_fct.call1(py, (x_py,))?;

        let fitvals: Vec<f64> = {
            #[cfg(feature = "numpy_support")]
            {
                if let Ok(arr) = fit_py.extract::<PyReadonlyArray1<f64>>(py) {
                    let slice = arr.as_slice()?;
                    slice.to_vec()
                } else {
                    fit_py.extract::<Vec<f64>>(py)?
                }
            }
            #[cfg(not(feature = "numpy_support"))]
            {
                fit_py.extract::<Vec<f64>>(py)?
            }
        };

        if fitvals.len() != x_candidates.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vectorized objective must return len(X) fitness values",
            ));
        }
        py.allow_threads(|| es.tell(x_candidates, fitvals));
        if disp_mod > 0 {
            py.allow_threads(|| es.disp(disp_mod));
        }
    }
    let (xbest, fbest, _evals_best, _counteval, _iters, xmean, _stds) = es.result();
    let xmean_val = {
        let x_py = PyList::new_bound(py, std::slice::from_ref(&xmean));
        let fit_py = objective_fct.call1(py, (x_py,))?;
        let vals: Vec<f64> = fit_py.extract(py)?;
        if vals.is_empty() {
            f64::INFINITY
        } else {
            vals[0]
        }
    };
    let xmin = if fbest < xmean_val {
        xbest.clone()
    } else {
        xmean.clone()
    };
    let es_py = Py::new(py, PyCmaes { inner: es })?;
    Ok((xmin, es_py))
}

fn apply_box_constraints(x: &mut [f64], lb: Option<&[f64]>, ub: Option<&[f64]>, mirror: bool) {
    if lb.is_none() && ub.is_none() {
        return;
    }
    let n = x.len();

    // Fast SIMD clamp path when mirroring is off.
    if !mirror {
        let mut i = 0;
        while i + SIMD_LANES <= n {
            let mut v = SimdF64::from_slice(&x[i..i + SIMD_LANES]);
            if let Some(l) = lb {
                let lvec =
                    SimdF64::from_slice(&l[i..i + SIMD_LANES.min(l.len().saturating_sub(i))]);
                v = v.simd_max(lvec);
            }
            if let Some(u) = ub {
                let uvec =
                    SimdF64::from_slice(&u[i..i + SIMD_LANES.min(u.len().saturating_sub(i))]);
                v = v.simd_min(uvec);
            }
            v.as_array()
                .iter()
                .zip(&mut x[i..i + SIMD_LANES])
                .for_each(|(s, t)| *t = *s);
            i += SIMD_LANES;
        }
        for j in i..n {
            if let Some(l) = lb {
                if j < l.len() && x[j] < l[j] {
                    x[j] = l[j];
                }
            }
            if let Some(u) = ub {
                if j < u.len() && x[j] > u[j] {
                    x[j] = u[j];
                }
            }
        }
        return;
    }

    // Mirror path (scalar but robust): reflect into [lb, ub] using modulo when width > 0.
    for i in 0..n {
        let lo = lb.and_then(|l| l.get(i));
        let hi = ub.and_then(|u| u.get(i));
        match (lo, hi) {
            (Some(&l), Some(&u)) if u > l => {
                let width = u - l;
                let delta = x[i] - l;
                let wrapped = (delta.rem_euclid(2.0 * width)).abs();
                x[i] = if wrapped <= width {
                    l + wrapped
                } else {
                    u - (wrapped - width)
                };
            }
            (Some(&l), _) if x[i] < l => x[i] = l + (l - x[i]).abs(),
            (_, Some(&u)) if x[i] > u => x[i] = u - (x[i] - u).abs(),
            _ => {}
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(
    signature = (objective_fct, xstart, sigma, constraints, maxfevals=None, ftarget=None, verb_disp=None, covariance_mode=None, noise=false)
)]
#[allow(clippy::too_many_arguments)]
fn fmin_constrained(
    py: Python<'_>,
    objective_fct: PyObject,
    xstart: Vec<f64>,
    sigma: f64,
    constraints: Bound<'_, PyDict>,
    maxfevals: Option<usize>,
    ftarget: Option<f64>,
    verb_disp: Option<usize>,
    covariance_mode: Option<&str>,
    noise: bool,
) -> PyResult<(Vec<f64>, Py<PyCmaes>)> {
    let mode = parse_covariance_mode(covariance_mode)?;
    let mut es = CmaesState::new(xstart, sigma, None, ftarget, maxfevals, mode);
    if noise {
        es = es.with_noise(NoiseConfig {
            threshold_rel: 1e-3,
            sigma_expand: 1.6,
            max_sigma_factor: 5.0,
            cooldown_iters: 5,
        });
    }
    let disp_mod = verb_disp.unwrap_or(100);

    let lb: Option<Vec<f64>> = match constraints.get_item("lower_bounds")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    let ub: Option<Vec<f64>> = match constraints.get_item("upper_bounds")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    let mirror: bool = constraints
        .get_item("mirror")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(false);
    let max_resamples: usize = constraints
        .get_item("max_resamples")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(3);
    let reject_fn: Option<PyObject> = constraints.get_item("reject")?.map(|v| v.into_py(py));
    let repair_fn: Option<PyObject> = constraints.get_item("repair")?.map(|v| v.into_py(py));
    let penalty_fn: Option<PyObject> = constraints.get_item("penalty")?.map(|v| v.into_py(py));

    loop {
        if es.has_terminated() {
            break;
        }
        let x_raw = py.allow_threads(|| es.ask());
        let mut x_eval: Vec<Vec<f64>> = Vec::with_capacity(x_raw.len());
        let mut fitvals: Vec<f64> = Vec::with_capacity(x_raw.len());
        for mut x in x_raw {
            // Box projection first.
            apply_box_constraints(&mut x, lb.as_deref(), ub.as_deref(), mirror);

            // Rejection/resample loop.
            let mut attempts = 0;
            if let Some(ref reject) = reject_fn {
                while attempts < max_resamples {
                    let row = PyList::new_bound(py, &x);
                    let feasible: bool = reject.call1(py, (row,))?.extract(py)?;
                    if feasible {
                        break;
                    }
                    attempts += 1;
                    // draw a new sample
                    x = es.ask_one();
                    apply_box_constraints(&mut x, lb.as_deref(), ub.as_deref(), mirror);
                }
            }

            if let Some(ref repair) = repair_fn {
                let row = PyList::new_bound(py, &x);
                let repaired = repair.call1(py, (row,))?;
                let new_x: Vec<f64> = repaired.extract(py)?;
                x = new_x;
            }
            let row = PyList::new_bound(py, &x);
            let val = objective_fct.call1(py, (row,))?;
            let mut f: f64 = val.extract(py)?;
            if let Some(ref pen) = penalty_fn {
                let row2 = PyList::new_bound(py, &x);
                let pval = pen.call1(py, (row2,))?;
                let p: f64 = pval.extract(py)?;
                f += p;
            } else if let Some(ref reject) = reject_fn {
                // If still infeasible and no penalty provided, assign a large penalty.
                let row = PyList::new_bound(py, &x);
                let feasible: bool = reject.call1(py, (row,))?.extract(py)?;
                if !feasible {
                    f += 1e6;
                }
            }
            x_eval.push(x);
            fitvals.push(f);
        }
        py.allow_threads(|| es.tell(x_eval, fitvals));
        if disp_mod > 0 {
            py.allow_threads(|| es.disp(disp_mod));
        }
    }
    let (xbest, fbest, _evals_best, _counteval, _iters, xmean, _stds) = es.result();
    let xmean_val = {
        let row = PyList::new_bound(py, &xmean);
        let val = objective_fct.call1(py, (row,))?;
        let f: f64 = val.extract(py)?;
        f
    };
    let xmin = if fbest < xmean_val {
        xbest.clone()
    } else {
        xmean.clone()
    };
    let es_py = Py::new(py, PyCmaes { inner: es })?;
    Ok((xmin, es_py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(
    signature = (objective_fct, xstart, sigma, max_total_fevals, ftarget=None, strategy=None, verb_disp=None, max_restarts=None, covariance_mode=None)
)]
fn fmin_restart(
    py: Python<'_>,
    objective_fct: PyObject,
    xstart: Vec<f64>,
    sigma: f64,
    max_total_fevals: usize,
    ftarget: Option<f64>,
    strategy: Option<&str>,
    verb_disp: Option<usize>,
    max_restarts: Option<usize>,
    covariance_mode: Option<&str>,
) -> PyResult<(Vec<f64>, Py<PyCmaes>)> {
    let mode = parse_covariance_mode(covariance_mode)?;
    let strat = strategy.unwrap_or("bipop").to_ascii_lowercase();
    let disp_mod = verb_disp.unwrap_or(100);

    let n = xstart.len();
    let base_params = CmaesParameters::new(n, None);
    let base_lam = base_params.lam;

    let mut total_evals: usize = 0;
    let mut best_x: Option<Vec<f64>> = None;
    let mut best_f = f64::INFINITY;
    let mut best_state: Option<CmaesState> = None;

    let max_restarts_val = max_restarts.unwrap_or(10);

    for restart in 0..max_restarts_val {
        if total_evals >= max_total_fevals {
            break;
        }
        let lam = match strat.as_str() {
            "ipop" => base_lam * (1usize << restart),
            "bipop" => {
                if restart % 2 == 0 {
                    base_lam * (1usize << (restart / 2))
                } else {
                    (4 + (3.0 * (n as f64).ln()) as usize).max(4)
                }
            }
            _ => base_lam * (1usize << restart),
        };

        let popsize = Some(lam);

        let restart_xstart = if let Some(ref bx) = best_x {
            let mut xs = bx.clone();
            let scale = sigma * ((restart + 1) as f64).sqrt();
            let mut rng = StdRng::from_entropy();
            for v in &mut xs {
                let s: f64 = rng.sample(StandardNormal);
                *v += scale * s;
            }
            xs
        } else {
            xstart.clone()
        };

        let remaining = max_total_fevals.saturating_sub(total_evals);
        if remaining == 0 {
            break;
        }
        let mut es = CmaesState::new(
            restart_xstart,
            sigma,
            popsize,
            ftarget,
            Some(remaining),
            mode,
        );

        loop {
            if es.has_terminated() {
                break;
            }
            let x_candidates = py.allow_threads(|| es.ask());
            let mut fitvals: Vec<f64> = Vec::with_capacity(x_candidates.len());
            for x in &x_candidates {
                let row = PyList::new_bound(py, x);
                let val = objective_fct.call1(py, (row,))?;
                let f: f64 = val.extract(py)?;
                fitvals.push(f);
            }
            py.allow_threads(|| es.tell(x_candidates, fitvals));
            if disp_mod > 0 {
                py.allow_threads(|| es.disp(disp_mod));
            }
            if es.counteval + total_evals >= max_total_fevals {
                break;
            }
        }

        total_evals += es.counteval;
        let (xbest_r, fbest_r, _evals_best, _counteval, _iters, _xmean_r, _stds_r) = es.result();
        if fbest_r < best_f {
            best_f = fbest_r;
            best_x = Some(xbest_r.clone());
            best_state = Some(es);
        }
        if best_f <= ftarget.unwrap_or(f64::NEG_INFINITY) {
            break;
        }
    }

    let final_state = if let Some(es) = best_state {
        es
    } else {
        CmaesState::new(
            xstart.clone(),
            sigma,
            Some(base_lam),
            ftarget,
            Some(max_total_fevals),
            mode,
        )
    };
    let (xbest, fbest, _evals_best, _counteval, _iters, xmean, _stds) = final_state.result();
    let xmin = if fbest < f64::INFINITY {
        xbest.clone()
    } else {
        xmean.clone()
    };
    let es_py = Py::new(py, PyCmaes { inner: final_state })?;
    Ok((xmin, es_py))
}

/// Minimize an objective function from pure Rust (no Python involved).
///
/// Returns the best solution vector and the final optimizer state.
pub fn optimize_rust<F>(
    xstart: Vec<f64>,
    sigma: f64,
    popsize: Option<usize>,
    maxfevals: Option<usize>,
    ftarget: Option<f64>,
    covariance_mode: CovarianceModeKind,
    objective: F,
) -> (Vec<f64>, CmaesState)
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    let mut es = CmaesState::new(xstart, sigma, popsize, ftarget, maxfevals, covariance_mode);
    loop {
        if es.has_terminated() {
            break;
        }
        let x_candidates = es.ask();
        let fitvals: Vec<f64> = x_candidates.par_iter().map(|x| objective(x)).collect();
        es.tell(x_candidates, fitvals);
    }
    let (xbest, _fbest, _evals_best, _counteval, _iters, _xmean, _stds) = es.result();
    (xbest, es)
}

/// Test-only helpers to run deterministic CMA-ES steps from integration tests.
#[cfg(any(test, feature = "test_utils"))]
pub mod test_utils {
    use super::{CmaesParameters, CmaesState, CovarianceModeKind, NoiseConfig};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rand_distr::StandardNormal;

    /// Run CMA-ES with a customizable RNG seed and covariance mode; returns best objective value.
    pub fn run_seeded_mode(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals: usize,
        ftarget: f64,
        seed: u64,
        covariance_mode: CovarianceModeKind,
        objective: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        let mut es = CmaesState::new_with_seed(
            x0,
            sigma,
            None,
            Some(ftarget),
            Some(maxfevals),
            covariance_mode,
            seed,
        );

        while !es.has_terminated() {
            let x_candidates = es.ask();
            let fitvals: Vec<f64> = x_candidates.iter().map(|x| objective(x)).collect();
            es.tell(x_candidates, fitvals);
        }

        let (_xbest, fbest, _, _, _, _, _) = es.result();
        fbest
    }

    /// Convenience wrapper: fixed seed (42) and full covariance.
    pub fn run_seeded(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals: usize,
        ftarget: f64,
        objective: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        run_seeded_mode(
            x0,
            sigma,
            maxfevals,
            ftarget,
            42,
            CovarianceModeKind::Full,
            objective,
        )
    }

    /// Seeded run with noise handling enabled (IPOP/BIPOP remains off).
    pub fn run_seeded_mode_noise(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals: usize,
        ftarget: f64,
        seed: u64,
        covariance_mode: CovarianceModeKind,
        objective: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        let cfg = NoiseConfig {
            threshold_rel: 1e-3,
            sigma_expand: 1.6,
            max_sigma_factor: 5.0,
            cooldown_iters: 5,
        };
        let mut es = CmaesState::new_with_seed(
            x0,
            sigma,
            None,
            Some(ftarget),
            Some(maxfevals),
            covariance_mode,
            seed,
        )
        .with_noise(cfg);

        while !es.has_terminated() {
            let x_candidates = es.ask();
            let fitvals: Vec<f64> = x_candidates.iter().map(|x| objective(x)).collect();
            es.tell(x_candidates, fitvals);
        }

        let (_xbest, fbest, _, _, _, _, _) = es.result();
        fbest
    }

    /// Run multiple seeds and return the best objective value.
    pub fn run_multiseed(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals: usize,
        ftarget: f64,
        seeds: &[u64],
        covariance_mode: CovarianceModeKind,
        objective: &dyn Fn(&[f64]) -> f64,
    ) -> f64 {
        let mut best = f64::INFINITY;
        for &seed in seeds {
            let f = run_seeded_mode(
                x0.clone(),
                sigma,
                maxfevals,
                ftarget,
                seed,
                covariance_mode,
                objective,
            );
            if f < best {
                best = f;
            }
        }
        best
    }

    /// Simple restart helper: perturb the start point between runs with Gaussian noise.
    pub fn run_with_restarts(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals_total: usize,
        ftarget: f64,
        restarts: usize,
        restart_sigma_scale: f64,
        seed: u64,
        covariance_mode: CovarianceModeKind,
        objective: &dyn Fn(&[f64]) -> f64,
    ) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let per_restart = (maxfevals_total / restarts.max(1)).max(1);
        let mut best = f64::INFINITY;
        for r in 0..restarts.max(1) {
            let mut xstart = x0.clone();
            if r > 0 {
                let scale = restart_sigma_scale * sigma * (r as f64).sqrt();
                for v in &mut xstart {
                    let n: f64 = rng.sample(StandardNormal);
                    *v += scale * n;
                }
            }
            let f = run_seeded_mode(
                xstart,
                sigma,
                per_restart,
                ftarget,
                seed + r as u64,
                covariance_mode,
                objective,
            );
            if f < best {
                best = f;
            }
        }
        best
    }

    /// Parallel-population restarts (IPOP/BIPOP) with adaptive lambda.
    /// Maintains multiple CMA-ES populations concurrently and shares a global
    /// evaluation budget. Returns the best objective value found.
    pub fn run_ipop_bipop_parallel(
        x0: Vec<f64>,
        sigma: f64,
        max_total_fevals: usize,
        ftarget: f64,
        max_restarts: usize,
        max_parallel: usize,
        seed: u64,
        covariance_mode: CovarianceModeKind,
        objective: &dyn Fn(&[f64]) -> f64,
    ) -> f64 {
        struct Pop {
            es: CmaesState,
            lam: usize,
        }

        if max_total_fevals == 0 || max_parallel == 0 {
            return f64::INFINITY;
        }

        let n = x0.len();
        let base_params = CmaesParameters::new(n, None);
        let base_lam = base_params.lam;

        let mut best = f64::INFINITY;
        let mut best_x: Option<Vec<f64>> = None;
        let mut remaining = max_total_fevals;
        let mut next_restart = 0usize;
        let mut pops: Vec<Pop> = Vec::new();

        while remaining > 0 && (!pops.is_empty() || next_restart < max_restarts) {
            while pops.len() < max_parallel && next_restart < max_restarts {
                let lam = if next_restart % 2 == 0 {
                    // IPOP branch grows lambda geometrically.
                    base_lam * (1usize << (next_restart / 2))
                } else {
                    // BIPOP branch keeps a small lambda capped to a few dims.
                    (4 + (3.0 * (n as f64).ln()) as usize).max(4)
                };

                let xstart = if let Some(ref bx) = best_x {
                    let mut xs = bx.clone();
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(next_restart as u64));
                    let scale = sigma * ((next_restart + 1) as f64).sqrt();
                    for v in &mut xs {
                        let n: f64 = rng.sample(StandardNormal);
                        *v += scale * n;
                    }
                    xs
                } else {
                    x0.clone()
                };

                // Ensure we respect remaining budget when instantiating.
                let es = CmaesState::new_with_seed(
                    xstart.clone(),
                    sigma,
                    Some(lam),
                    Some(ftarget),
                    Some(remaining),
                    covariance_mode,
                    seed.wrapping_add(next_restart as u64),
                );

                // Guard against tiny remaining budget: if lambda exceeds remaining, skip.
                if remaining < lam {
                    break;
                }

                // Pre-update best with initial state mean if helpful.
                let (_xbest, fbest, _, _, _, _xmean, _) = es.result();
                if fbest < best {
                    best = fbest;
                    best_x = Some(_xbest.clone());
                }

                pops.push(Pop { es, lam });
                next_restart += 1;
            }

            if pops.is_empty() {
                break;
            }

            let mut idx = 0;
            while idx < pops.len() {
                let lam = pops[idx].lam;
                if remaining < lam {
                    pops.swap_remove(idx);
                    continue;
                }
                let x_candidates = pops[idx].es.ask();
                let fitvals: Vec<f64> = x_candidates.iter().map(|x| objective(x)).collect();
                pops[idx].es.tell(x_candidates, fitvals);
                remaining = remaining.saturating_sub(lam);

                let (xbest, fbest, _, _, _, _, _) = pops[idx].es.result();
                if fbest < best {
                    best = fbest;
                    best_x = Some(xbest.clone());
                }

                if pops[idx].es.has_terminated() {
                    pops.swap_remove(idx);
                } else {
                    idx += 1;
                }
            }
        }

        best
    }

    /// Augmented Lagrangian penalty helper for inequality constraints g_i(x) <= 0.
    pub fn augmented_lagrangian_penalty_raw(g: &[f64], lambda: &[f64], rho: f64) -> f64 {
        let mut pen = 0.0;
        for (gi, li) in g.iter().zip(lambda.iter()) {
            let pos = gi.max(0.0);
            pen += li * pos + 0.5 * rho * pos * pos;
        }
        pen
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn fastcma(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCmaes>()?;
    m.add_class::<PyFF>()?;
    m.add_function(wrap_pyfunction!(fmin, m)?)?;
    m.add_function(wrap_pyfunction!(fmin_vec, m)?)?;
    m.add_function(wrap_pyfunction!(fmin_constrained, m)?)?;
    m.add_function(wrap_pyfunction!(fmin_restart, m)?)?;
    let ff_obj = Py::new(py, PyFF)?;
    m.add("ff", ff_obj)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_basic_optimization(
        x0: Vec<f64>,
        sigma: f64,
        maxfevals: usize,
        ftarget: f64,
        objective: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        let mut es = CmaesState::new_with_seed(
            x0,
            sigma,
            None,
            Some(ftarget),
            Some(maxfevals),
            CovarianceModeKind::Full,
            42,
        );

        while !es.has_terminated() {
            let x_candidates = es.ask();
            let fitvals: Vec<f64> = x_candidates.iter().map(|x| objective(x)).collect();
            es.tell(x_candidates, fitvals);
        }

        let (_xbest, fbest, _, _, _, _, _) = es.result();
        fbest
    }

    #[test]
    fn sphere_reaches_near_zero() {
        let fbest = run_basic_optimization(vec![0.8; 6], 0.4, 4000, 1e-12, |x| {
            x.iter().map(|v| v * v).sum::<f64>()
        });
        assert!(fbest < 1e-8, "sphere optimum not reached: {fbest}");
    }

    #[test]
    fn rosenbrock_2d_reaches_minimum() {
        let fbest = run_basic_optimization(vec![-1.2, 1.0], 0.5, 20000, 1e-10, |x| {
            // classic 2D Rosenbrock; minimum at (1,1) with f=0
            let x0 = x[0];
            let x1 = x[1];
            100.0 * (x0 * x0 - x1).powi(2) + (x0 - 1.0).powi(2)
        });
        assert!(fbest < 1e-4, "rosenbrock optimum not reached: {fbest}");
    }
}
