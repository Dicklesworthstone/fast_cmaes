<div align="center">

# ⚡ fastcma

**Hyper-optimized CMA-ES in Rust with a first-class Python experience**

[![CI](https://img.shields.io/github/actions/workflow/status/Dicklesworthstone/fast_cmaes/build-wheels.yml?branch=main&label=CI&logo=github&style=for-the-badge)](https://github.com/Dicklesworthstone/fast_cmaes/actions/workflows/build-wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/fast-cmaes?label=PyPI&logo=pypi&logoColor=white&style=for-the-badge&color=3776ab)](https://pypi.org/project/fast-cmaes/)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-3776ab?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-nightly-ce412b?logo=rust&logoColor=white&style=for-the-badge)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

**SIMD** • **rayon** • **deterministic seeds** • **vectorized objectives** • **restarts** • **constraints** • **Rich TUI**

Published to PyPI as [`fast-cmaes`](https://pypi.org/project/fast-cmaes/) (module name: `fastcma`). Latest release: **v0.1.4**

</div>

## 📑 Table of Contents

- [Why CMA-ES](#-why-cma-es)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation-python)
- [Quickstart](#-quickstart-python)
- [Vectorized & Constraints](#-vectorized--constraints)
- [Rust Usage](#-rust-usage-library)
- [C/C++ FFI](#-cc-ffi)
- [REST API](#-rest-api-example)
- [Demos & TUI](#-demos--visualization)
- [Baselines & Benchmarks](#-baselines--benchmarks)
- [Performance](#-performance-considerations)
- [Feature Flags](#-feature-flags)
- [Testing](#-testing)
- [License](#-license)

## 🎯 Why CMA-ES

<div align="center">

**Derivative-free** • **Adaptive** • **Parallel-friendly**

</div>

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) represents a paradigm shift in optimization. Invented by Nikolaus Hansen in 1996, CMA-ES emerged from decades of research into evolution strategies and has since become one of the most successful derivative-free optimization algorithms. Hansen, who continues to research and refine CMA-ES along with other evolutionary algorithms, developed the method to address fundamental limitations in existing optimization techniques. Unlike gradient-based methods that require smooth, differentiable objectives, CMA-ES thrives on black-box problems where gradients are unavailable, noisy, or meaningless. The algorithm automatically adapts its search distribution by learning the problem's structure through successful search directions, making it particularly effective for non-convex landscapes with curved valleys and deceptive local minima.

The core strength lies in its adaptive mechanisms. The step size $$\sigma$$ controls exploration radius, while the covariance matrix $$\mathbf{C}$$ shapes the search distribution to follow favorable directions. This dual adaptation allows CMA-ES to navigate complex fitness landscapes that would trap simpler optimizers. Moreover, the population-based approach means candidate evaluations are embarrassingly parallel, enabling near-linear speedup with additional CPU cores. Over nearly three decades of development, CMA-ES has evolved from a research prototype into a mature, production-ready algorithm with proven effectiveness across diverse application domains.

**Key Advantages:**

| Feature | Benefit |
|---------|---------|
| **No gradients required** | Works with black-box, noisy, or discontinuous objectives |
| **Adaptive step size** | Automatically balances exploration vs exploitation |
| **Covariance adaptation** | Learns problem structure and follows favorable search directions |
| **Population-based** | Naturally parallelizable, robust to local minima |

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph PythonAPI["🐍 Python API Layer"]
        direction TB
        A["✨ fmin<br/>fmin_vec<br/>CMAES class"]
        B["🔒 Constraints<br/>and restarts"]
        C["📊 Naive baseline<br/>pure Python"]
        A -.->|uses| B
        C -.->|compares| A
    end

    subgraph RustCore["🦀 Rust Core Engine<br/>src/lib.rs"]
        direction TB
        D["🔄 Ask Tell loop<br/>Core optimizer"]
        E["📐 Covariance update<br/>full or diagonal"]
        F["📈 Sigma adaptation<br/>step size control"]
        G["⚡ SIMD dot products<br/>rayon parallel fitness"]
        H["🎲 Deterministic seeds<br/>reproducible runs"]
        D -->|updates| E
        E -->|adapts| F
        F -->|feeds back| D
        D -->|uses| G
        D -->|seeds| H
    end

    subgraph TestsDemos["🧪 Tests & Demos"]
        direction TB
        I["📉 Benchmarks<br/>sphere rosenbrock<br/>rastrigin ackley"]
        J["🎨 Rich TUI demo<br/>live visualization"]
        K["✅ Python smoke<br/>integration tests"]
    end

    PythonAPI -->|"API calls"| RustCore
    TestsDemos -->|"validates"| RustCore
    TestsDemos -->|"demonstrates"| PythonAPI

    classDef pythonStyle fill:#3776ab,stroke:#ffd43b,stroke-width:3px,color:#fff,font-weight:bold
    classDef rustStyle fill:#ce412b,stroke:#fbbf24,stroke-width:3px,color:#fff,font-weight:bold
    classDef testStyle fill:#10b981,stroke:#34d399,stroke-width:3px,color:#fff,font-weight:bold
    classDef subgraphStyle fill:#f9fafb,stroke:#374151,stroke-width:3px

    class A,B,C pythonStyle
    class D,E,F,G,H rustStyle
    class I,J,K testStyle
    class PythonAPI,RustCore,TestsDemos subgraphStyle
```

### Architecture Deep Dive

#### Core Components

**1. Ask-Tell Interface (`CmaesState`)**

The optimizer follows the classic CMA-ES ask-tell pattern:

```mermaid
flowchart TD
    Start([Start Optimization]) --> Init[Initialize xmean, sigma, C]
    Init --> Ask[ask: Generate λ candidates]
    Ask --> Sample[Sample from Nxmean, sigma²C]
    Sample --> Eval[Evaluate fitness fxi]
    Eval --> Tell[tell: Update distribution]
    Tell --> Rank[Rank candidates by fitness]
    Rank --> UpdateMean[Update xmean weighted mean]
    UpdateMean --> UpdatePaths[Update evolution paths pc, ps]
    UpdatePaths --> UpdateCov[Update covariance C]
    UpdateCov --> UpdateSigma[Adapt step size sigma]
    UpdateSigma --> CheckTerm{Terminated?}
    CheckTerm -->|No| Ask
    CheckTerm -->|Yes| End([Return best solution])
    
    classDef askStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef tellStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef decisionStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef startEndStyle fill:#8b5cf6,stroke:#6d28d9,stroke-width:3px,color:#fff
    
    class Ask,Sample,Eval askStyle
    class Tell,Rank,UpdateMean,UpdatePaths,UpdateCov,UpdateSigma tellStyle
    class CheckTerm decisionStyle
    class Start,End startEndStyle
```

**Mathematical Formulation:**

- **`ask()`**: Generates $$\lambda$$ candidate solutions by sampling:
  
  $$x_i \sim \mathcal{N}(\mathbf{x}_{\mathrm{mean}}, \sigma^2 \mathbf{C})$$
  
  where $$\mathbf{C}$$ is the covariance matrix and $$\sigma$$ is the step size.

- **`tell()`**: Updates distribution parameters:
  - **Weighted mean**: 
    
    $$\mathbf{x}_{\mathrm{mean}} \leftarrow \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}$$
  
  - **Evolution paths**: 
    
    $$\mathbf{p}_c \leftarrow (1-c_c)\mathbf{p}_c + c_c h_{\sigma} \frac{\mathbf{x}_{\mathrm{mean}} - \mathbf{x}_{\mathrm{old}}}{\sigma}$$
    
    $$\mathbf{p}_{\sigma} \leftarrow (1-c_{\sigma})\mathbf{p}_{\sigma} + c_{\sigma} \frac{\mathbf{C}^{-1/2}(\mathbf{x}_{\mathrm{mean}} - \mathbf{x}_{\mathrm{old}})}{\sigma}$$
  
  - **Covariance update**: 
    
    $$\mathbf{C} \leftarrow (1-c_1-c_{\mu})\mathbf{C} + c_1 \mathbf{p}_c \mathbf{p}_c^T + c_{\mu} \sum_{i=1}^{\mu} w_i \mathbf{y}_{i:\lambda} \mathbf{y}_{i:\lambda}^T$$
  
  - **Step size adaptation**: 
    
    $$\sigma \leftarrow \sigma \exp\left(\frac{c_{\sigma}}{d_{\sigma}}\left(\frac{\|\mathbf{p}_{\sigma}\|}{\mathbb{E}\|\mathcal{N}(\mathbf{0},\mathbf{I})\|} - 1\right)\right)$$

**2. Covariance Representation (`CovarianceMode`)**

Two modes optimize for different problem characteristics:

```mermaid
flowchart TD
    Start([Covariance Mode Selection]) --> CheckDim{Dimension n}
    CheckDim -->|n < 50| Full[Full Covariance<br/>n×n matrix<br/>O n³ eigen decomp]
    CheckDim -->|n ≥ 50 or separable| Diag[Diagonal Covariance<br/>n elements<br/>O n updates]
    
    Full --> FullFeatures[Captures correlations<br/>Adapts to curved valleys<br/>Slower but more accurate]
    Diag --> DiagFeatures[Assumes independence<br/>Faster updates<br/>Good for separable problems]
    
    FullFeatures --> FullUse[Best for:<br/>Low-medium dimensions<br/>Correlated variables<br/>Complex landscapes]
    DiagFeatures --> DiagUse[Best for:<br/>High dimensions<br/>Separable problems<br/>Speed-critical]
    
    classDef fullStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef diagStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef decisionStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    
    class Full,FullFeatures,FullUse fullStyle
    class Diag,DiagFeatures,DiagUse diagStyle
    class CheckDim decisionStyle
```

**Mathematical Formulation:**

The choice between full and diagonal covariance represents a fundamental trade-off between modeling accuracy and computational cost. Full covariance stores the complete $$n \times n$$ matrix $$\mathbf{C}$$, enabling adaptation to arbitrary search distributions and capturing correlations between variables. This comes at the cost of $$O(n^3)$$ eigen decomposition for sampling, making it best suited for low-to-medium dimensions where correlations significantly impact optimization.

Diagonal covariance takes a pragmatic approach, storing only the $$n$$ diagonal elements and assuming variable independence. This reduces storage from $$O(n^2)$$ to $$O(n)$$ and eliminates the expensive eigen decomposition, trading correlation modeling for speed. The diagonal mode excels on separable problems where variables can be optimized independently, or in high dimensions where full covariance becomes computationally prohibitive.

**Covariance Mode Comparison:**

| Aspect | Full Covariance | Diagonal Covariance |
|--------|----------------|---------------------|
| **Storage** | $$O(n^2)$$ | $$O(n)$$ |
| **Sampling cost** | $$O(n^3)$$ eigen decomposition | $$O(n)$$ element-wise |
| **Correlation modeling** | Captures all pairwise correlations | Assumes independence |
| **Best for** | Low-medium dimensions (< 50), correlated variables | High dimensions (> 50), separable problems |
| **Convergence quality** | Higher (adapts to problem structure) | Lower (but often sufficient) |
| **Speed** | Slower (eigen decomposition bottleneck) | Faster (linear scaling) |

**Covariance Update Process:**

The covariance matrix evolves through a combination of rank-one and rank-μ updates:

$$\mathbf{C} \leftarrow (1-c_1-c_{\mu})\mathbf{C} + c_1 \mathbf{p}_c \mathbf{p}_c^T + c_{\mu} \sum_{i=1}^{\mu} w_i \mathbf{y}_{i:\lambda} \mathbf{y}_{i:\lambda}^T$$

The first term decays the current covariance, preventing it from growing unbounded. The second term incorporates the evolution path $$\mathbf{p}_c$$, capturing the direction of recent successful steps. The third term aggregates information from the current population, with weights $$w_i$$ emphasizing better solutions. This dual mechanism combines short-term momentum (evolution path) with long-term learning (population statistics), enabling robust adaptation to diverse problem landscapes.

**3. Parameter Adaptation**
- **Cumulative step-size adaptation**: Evolution path `ps` tracks the direction of recent steps, enabling faster convergence along favorable directions
- **Cumulative covariance adaptation**: Evolution path `pc` accumulates successful search directions, shaping the covariance matrix
- **Rank-μ and rank-one updates**: Combines information from the current population (rank-μ) and the evolution path (rank-one) for robust covariance adaptation

#### Performance Optimizations

**Optimization Strategy Overview:**

```mermaid
flowchart TD
    Start([Performance Optimization]) --> SIMD[SIMD Acceleration<br/>4x speedup on dot products]
    Start --> Lazy[Lazy Eigensystem<br/>5-10x fewer decompositions]
    Start --> Rayon[Rayon Parallelization<br/>Linear speedup with cores]
    Start --> Buffers[Pre-allocated Buffers<br/>Zero-allocation loops]
    Start --> Symmetry[Symmetry Enforcement<br/>Numerical stability]
    
    SIMD --> SIMDImpact[Used in:<br/>Mahalanobis norm<br/>Square sums<br/>Constraint clamping]
    Lazy --> LazyImpact[Critical for:<br/>High dimensions n > 20<br/>Reduces O n³ cost]
    Rayon --> RayonImpact[Best for:<br/>Expensive objectives<br/>Parallelizable workloads]
    Buffers --> BufferImpact[Critical for:<br/>Low-latency apps<br/>Hot path optimization]
    Symmetry --> SymImpact[Applied before:<br/>Eigen decomposition<br/>Prevents numerical drift]
    
    classDef simdStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef lazyStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef rayonStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef bufferStyle fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    classDef symStyle fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
    
    class SIMD,SIMDImpact simdStyle
    class Lazy,LazyImpact lazyStyle
    class Rayon,RayonImpact rayonStyle
    class Buffers,BufferImpact bufferStyle
    class Symmetry,SymImpact symStyle
```

**1. SIMD-Accelerated Dot Products**
```rust
// Uses portable_simd with 4 f64 lanes (256-bit AVX on x86_64)
fn dot_simd(a: &[f64], b: &[f64]) -> f64
```
- Processes 4 elements per iteration using SIMD instructions
- Falls back to scalar for remainder elements
- Used in Mahalanobis norm computation, square sums, and constraint clamping
- **Impact**: ~3-4x speedup on dot products compared to naive scalar loops

**2. Lazy Eigensystem Updates**

The most expensive operation ($$O(n^3)$$ eigen decomposition) is deferred using an adaptive gap:

$$\mathrm{lazy\_gap\_evals} = \frac{0.5 \cdot n \cdot \lambda}{(c_1 + c_{\mu}) \cdot n^2}$$

```mermaid
flowchart TD
    AskCall["ask called"] --> CheckGap{"current_eval > updated_eval + gap?"}
    CheckGap -->|No| UseCache["Use cached eigenbasis"]
    CheckGap -->|Yes| EnforceSym["Enforce symmetry A = (A + A^T) / 2"]
    EnforceSym --> EigenDecomp["Eigen decomposition O(n³)"]
    EigenDecomp --> FloorEigs["Floor eigenvalues to 1e-20"]
    FloorEigs --> ComputeInvSqrt["Compute C^(-1/2)"]
    ComputeInvSqrt --> UpdateCache["Update cached eigenbasis"]
    UpdateCache --> UseCache
    UseCache --> Sample["Sample candidates"]
    
    classDef decisionStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef computeStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef cacheStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    
    class CheckGap decisionStyle
    class EnforceSym,EigenDecomp,FloorEigs,ComputeInvSqrt,UpdateCache computeStyle
    class UseCache,Sample cacheStyle
```

- Eigensystem only recomputed when $$\mathrm{current\_eval} > \mathrm{updated\_eval} + \mathrm{lazy\_gap\_evals}$$
- Gap grows with dimension and learning rates, naturally reducing update frequency
- Covariance matrix updates continue (cheap rank-one/rank-μ), but sampling uses cached eigenbasis
- **Impact**: Reduces eigen decompositions by 5-10x in typical runs, critical for high-dimensional problems

**3. Rayon Parallelization**
- **Fitness evaluation**: `x_candidates.par_iter().map(|x| objective(x))` parallelizes embarrassingly parallel objective calls
- **Covariance updates**: `data.par_chunks_mut(n).for_each(...)` parallelizes outer product accumulation
- **Impact**: Near-linear speedup with CPU cores (e.g., 8 cores → ~7x faster on parallelizable objectives)

**4. Pre-allocated Buffers**
Reusable scratch buffers (`z_buf`, `y_buf`, `tmp_buf`) eliminate allocations in hot paths:
- `z_buf`: Standard normal samples in eigenbasis space
- `y_buf`: Transformed samples in original space
- `tmp_buf`: Temporary storage for matrix-vector operations
- **Impact**: Zero-allocation inner loops, critical for low-latency applications

**5. Symmetry Enforcement**
Covariance matrices must be symmetric, but floating-point updates can introduce asymmetry:
```rust
fn enforce_symmetry(&mut self) // Averages A[i,j] and A[j,i]
```
- Ensures numerical stability without expensive checks on every access
- Applied only before eigen decomposition (lazy evaluation)

#### Robustness & Reliability Features

Robustness in fastcma extends beyond handling difficult optimization problems to ensuring reliable behavior under diverse conditions. The implementation includes multiple layers of protection against common failure modes: numerical instability, noisy evaluations, invalid inputs, and infinite loops. These features work together to create a system that gracefully handles edge cases rather than crashing or producing incorrect results.

**1. Deterministic Seeding**
```rust
fn new_with_seed(..., seed: u64) -> Self {
    state.rng = StdRng::seed_from_u64(seed);
}
```
- **Why**: Enables reproducible benchmarks, regression testing, and debugging
- **Implementation**: Uses `rand::StdRng` with `SeedableRng` trait
- **Impact**: Same seed → identical optimization trajectory (critical for CI/CD and scientific reproducibility)

**2. Noise Handling**
Detects when fitness evaluations are too similar (indicating noise or stagnation):
```rust
if spread < threshold_rel {
    sigma *= sigma_expand; // Expand step size to escape
    noise_cooldown = cooldown_iters; // Prevent oscillation
}
```
- **Why**: Noisy objectives can trap CMA-ES with artificially small step sizes
- **Detection**: Compares fitness spread (`mid - best`) relative to best value
- **Response**: Temporarily expands `sigma` with cooldown to prevent thrashing
- **Impact**: Robust performance on noisy objectives (simulations, Monte Carlo, etc.)

**3. Eigenvalue Flooring**
Prevents numerical issues from negative or near-zero eigenvalues:
```rust
if min_ev <= 0.0 {
    for ev in &mut eigs {
        if *ev < eps { *ev = eps; } // Floor to 1e-20
    }
}
```
- **Why**: Negative eigenvalues break sampling; near-zero cause numerical instability
- **Impact**: Prevents panics and ensures valid covariance matrices even under numerical drift

**4. Constraint Handling Pipeline**

Multi-stage constraint satisfaction with fallback mechanisms:

```mermaid
flowchart TD
    Sample[Sample candidate x] --> BoxProj[Box Projection<br/>SIMD clamp to lb, ub]
    BoxProj --> Mirror{mirror enabled?}
    Mirror -->|Yes| MirrorOp[Reflect into bounds<br/>using modulo]
    Mirror -->|No| CheckReject{reject function?}
    MirrorOp --> CheckReject
    CheckReject -->|Yes| RejectCheck[Check feasibility]
    RejectCheck -->|Infeasible| Resample{attempts < max_resamples?}
    Resample -->|Yes| Sample
    Resample -->|No| CheckRepair{repair function?}
    RejectCheck -->|Feasible| CheckRepair
    CheckReject -->|No| CheckRepair
    CheckRepair -->|Yes| RepairOp[Repair x]
    CheckRepair -->|No| EvalObj[Evaluate objective fx]
    RepairOp --> EvalObj
    EvalObj --> CheckPenalty{penalty function?}
    CheckPenalty -->|Yes| AddPenalty[Add penalty to f]
    CheckPenalty -->|No| CheckRejectFinal{reject function<br/>still infeasible?}
    CheckRejectFinal -->|Yes| LargePenalty[Add large penalty 1e6]
    CheckRejectFinal -->|No| ReturnFit[Return fitness]
    AddPenalty --> ReturnFit
    LargePenalty --> ReturnFit
    
    classDef sampleStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef constraintStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef decisionStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef penaltyStyle fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
    
    class Sample,Resample sampleStyle
    class BoxProj,MirrorOp,RejectCheck,RepairOp constraintStyle
    class Mirror,CheckReject,Resample,CheckRepair,CheckPenalty,CheckRejectFinal decisionStyle
    class AddPenalty,LargePenalty penaltyStyle
    class EvalObj,ReturnFit sampleStyle
```

**Stages:**

1. **Box projection** (SIMD-accelerated): Clamps to $$[\mathbf{lb}, \mathbf{ub}]$$ bounds
   
   $$x_i \leftarrow \max(\min(x_i, ub_i), lb_i)$$

2. **Mirroring** (optional): Reflects out-of-bounds points into feasible region
   
   $$x_i \leftarrow \begin{cases} lb_i + |x_i - lb_i| \bmod (2w_i) & \mathrm{if\ } |x_i - lb_i| \bmod (2w_i) \leq w_i \\ ub_i - (|x_i - lb_i| \bmod (2w_i) - w_i) & \mathrm{otherwise} \end{cases}$$
   
   where $$w_i = ub_i - lb_i$$

3. **Rejection/resampling**: Up to `max_resamples` attempts if `reject()` predicate fails

4. **Repair** (optional): User-provided function to fix infeasible points

5. **Penalty** (optional): Adds penalty to fitness for remaining violations
   
   $$f_{\mathrm{penalized}} = f(\mathbf{x}) + \mathrm{penalty}(\mathbf{x})$$

- **Why**: Flexible constraint handling accommodates diverse problem types
- **Impact**: Works with box constraints, nonlinear constraints (via penalty), and custom feasibility rules

**5. NaN Handling**
Robust sorting handles NaN fitness values:
```rust
match (ai.partial_cmp(&aj), ai.is_nan(), aj.is_nan()) {
    (_, true, false) => Ordering::Greater,  // NaN > valid
    (_, false, true) => Ordering::Less,     // valid < NaN
    ...
}
```
- **Why**: Invalid objective evaluations shouldn't crash the optimizer
- **Impact**: Graceful degradation when objectives return NaN/Inf

**6. Termination Conditions**
Multiple termination criteria prevent infinite loops:
- **Max evaluations**: Hard budget limit
- **Target fitness**: Early exit on convergence
- **Condition number**: Stops if covariance becomes ill-conditioned (> 1e14)
- **TolFun**: Stops if fitness improvement stalls (< 1e-12)
- **TolX**: Stops if step size becomes negligible
- **Impact**: Reliable termination under diverse conditions

#### Design Decisions & Trade-offs

Every design decision in fastcma reflects careful consideration of trade-offs between performance, usability, and maintainability. The choices prioritize practical benefits for end users while maintaining code quality and extensibility. These decisions aren't arbitrary but represent informed choices based on profiling, benchmarking, and real-world usage patterns.

**1. Rust + PyO3 vs Pure Python**

The choice to implement the core in Rust with Python bindings represents a fundamental trade-off between development complexity and runtime performance. Rust provides zero-cost abstractions that enable high-level code without runtime overhead, memory safety that prevents entire classes of bugs, and direct access to SIMD instructions for vectorization. PyO3 bridges the gap between Rust's performance and Python's ecosystem, enabling seamless integration with minimal overhead.

The trade-off manifests in build complexity: users must compile Rust code rather than simply installing Python packages. However, the performance gains justify this complexity: core operations achieve 10-100x speedups compared to pure Python implementations. The inclusion of a pure Python baseline (`fastcma_baseline`) enables direct performance comparisons, demonstrating the value of the Rust implementation.

**2. Full vs Diagonal Covariance**

Rather than forcing a single covariance representation, fastcma provides user-selectable modes that optimize for different problem characteristics. Full covariance captures all pairwise variable correlations, enabling adaptation to arbitrary search distributions, but requires $$O(n^3)$$ eigen decomposition for sampling. Diagonal covariance assumes variable independence, reducing storage to $$O(n)$$ and eliminating expensive decompositions, but sacrifices correlation modeling.

The trade-off becomes clear in high dimensions: full covariance provides better convergence quality but becomes computationally prohibitive, while diagonal covariance offers acceptable quality with dramatically better performance. By making this configurable, users can choose based on their specific problem characteristics and computational constraints.

**3. Lazy vs Eager Eigensystem Updates**

The lazy eigensystem update strategy defers expensive $$O(n^3)$$ eigen decompositions until necessary, rather than recomputing every iteration. This decision recognizes that covariance matrix updates (cheap rank-one and rank-μ operations) can proceed without immediate eigen decomposition, and the eigenbasis remains sufficiently accurate for several iterations.

The trade-off involves slightly stale eigenbasis information between updates, but the performance gain is substantial: 5-10x fewer decompositions in typical runs. The adaptive gap calculation ensures updates occur frequently enough to maintain accuracy while avoiding unnecessary computation. This optimization becomes critical for high-dimensional problems where eigen decomposition dominates runtime.

**4. Rayon vs Sequential**

The choice to use Rayon for parallelization reflects the embarrassingly parallel nature of fitness evaluation. Unlike some algorithms where parallelization introduces complexity or synchronization overhead, CMA-ES naturally parallelizes: each candidate can be evaluated independently. Rayon provides a zero-cost abstraction that automatically scales to available CPU cores.

The trade-off involves slight thread pool overhead, but this is negligible compared to the linear speedup achieved on parallelizable objectives. For expensive objective functions that dominate runtime, parallelization can reduce wall-clock time by factors matching core count, making this an essential optimization for practical applications.

**5. Deterministic vs Non-deterministic**

Deterministic seeding by default prioritizes reproducibility over slight performance gains from faster random number generators. This choice recognizes that reproducibility is essential for debugging, regression testing, and scientific validation. The performance cost is negligible: modern RNGs are fast enough that seeding overhead doesn't impact optimization runtime.

The trade-off enables reproducible benchmarks, making it possible to verify that optimizations don't regress performance and that bug fixes actually resolve issues. This determinism is critical for CI/CD pipelines and scientific reproducibility, where identical inputs must produce identical outputs.

## ✨ Features

<div align="center">

**Production-ready** • **High-performance** • **Developer-friendly**

</div>

Fastcma delivers a comprehensive optimization toolkit that balances ease of use with raw performance. The Python API provides multiple interfaces tailored to different use cases, from simple one-line optimizations to fine-grained control over the optimization process. Under the hood, a carefully optimized Rust core leverages modern CPU features like SIMD instructions and multi-core parallelism to achieve performance that rivals or exceeds hand-tuned C implementations.

### 🐍 Python API

The API design follows a progressive disclosure principle. New users can start with `fmin` for immediate results, while advanced users can leverage `fmin_vec` for batch processing, `fmin_constrained` for complex constraint handling, or the full `CMAES` class for complete control. The `fmin_restart` function implements proven IPOP and BIPOP strategies that automatically escape local minima by adaptively increasing population size.

| Function | Use Case | Complexity |
|----------|----------|------------|
| **`fmin`** | Quick optimization with minimal setup | Low |
| **`fmin_vec`** | Batch evaluation for vectorized objectives | Medium |
| **`fmin_constrained`** | Box constraints, nonlinear constraints, custom feasibility | Medium |
| **`fmin_restart`** | Escaping local minima with adaptive restarts | Medium |
| **`CMAES` class** | Full ask-tell control for custom workflows | High |

### ⚡ Performance

Performance optimization follows a multi-layered strategy. SIMD acceleration targets the most frequent operations like dot products, achieving 3-4x speedups on modern CPUs. Rayon parallelization exploits the embarrassingly parallel nature of fitness evaluation, scaling linearly with core count. The lazy eigensystem update defers expensive $$O(n^3)$$ eigen decompositions until necessary, reducing their frequency by 5-10x. Pre-allocated buffers eliminate memory allocations in hot paths, critical for low-latency applications.

| Optimization | Technique | Speedup | When It Matters |
|--------------|-----------|---------|-----------------|
| **SIMD** | Vectorized dot products | 3-4x | High-dimensional problems |
| **Rayon** | Parallel fitness evaluation | Linear with cores | Expensive objectives |
| **Lazy eigen** | Deferred decomposition | 5-10x fewer ops | Dimensions > 20 |
| **Pre-alloc buffers** | Zero-allocation loops | Eliminates GC pauses | Low-latency apps |

### 📐 Flexibility

The optimizer adapts to diverse problem characteristics through configurable modes and automatic adaptations. Users can choose between full covariance for capturing correlations or diagonal covariance for speed in high dimensions. Deterministic seeding enables reproducible research and debugging. Noise handling automatically detects and escapes stagnation caused by noisy evaluations. The constraint handling pipeline provides multiple fallback mechanisms, from simple box projection to sophisticated penalty methods.

### 🛠️ Developer Experience

Developer experience receives equal attention to performance. A pure-Python baseline (`fastcma.cma_es`) enables direct speed comparisons. The Rich TUI provides real-time visualization of optimization progress. Cross-platform wheels eliminate build complexity for end users. C/C++ FFI headers enable native integration, while the REST API example demonstrates remote optimization workflows.

## 📦 Installation (Python)

Installation follows standard Python package management practices, with pre-built wheels available for most platforms. The PyPI package includes optimized binaries compiled for common architectures, eliminating the need for Rust toolchain installation for most users. For developers modifying the Rust code, source builds enable full control over compilation options and feature flags.

### Quick Install (PyPI)

The simplest installation path uses PyPI's pre-built wheels, which work out of the box on Linux, macOS, and Windows for Python 3.12 through 3.14. The package installs as the `fastcma` module, providing immediate access to all optimization functions.

```bash
python -m pip install fast-cmaes==0.1.4  # installs module `fastcma`
```

### Build from Source

Source builds become necessary when modifying Rust code, enabling specific feature flags, or targeting platforms without pre-built wheels. The build process uses maturin, which handles the complexity of compiling Rust extensions and generating Python-compatible wheels. Optional features like NumPy support and LAPACK backends require additional dependencies but provide performance benefits for specific use cases.

Build locally (needed only if hacking on Rust):

```bash
python -m pip install maturin
maturin develop --release

# Optional: NumPy fast paths
maturin develop --release --features numpy_support

# Optional: LAPACK eigen backend (requires a Fortran toolchain)
maturin develop --release --features eigen_lapack

# Demo extras (Rich TUI)
python -m pip install .[demo]
```

### 🚀 One-Liner Setup

One-liner setup + Rich TUI demo (auto-installs nightly Rust, uv, venv, builds, runs):

```bash
./scripts/setup_and_demo.sh
```

## 🚀 Quickstart (Python)

### Basic Usage

```python
from fastcma import fmin
from fastcma_baseline import benchmark_sphere

def sphere(x):
    return sum(v*v for v in x)

xmin, es = fmin(sphere, [0.5, -0.2, 0.8], sigma=0.3, maxfevals=4000, ftarget=1e-12)
print("xmin", xmin)

# Pure-Python baseline
print(benchmark_sphere(dim=20, iters=120))
```

## 🔄 Vectorized & Constraints

### Vectorized Evaluation

```python
from fastcma import fmin_vec

def sphere_vec(X):
    return [sum(v*v for v in x) for x in X]

xmin, _ = fmin_vec(sphere_vec, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=3000)
```

### Constrained Optimization

```python
from fastcma import fmin_constrained

def sphere(x): 
    return sum(v*v for v in x)

constraints = {"lower_bounds": [-1, -1, -1], "upper_bounds": [1, 1, 1]}
xmin, _ = fmin_constrained(sphere, [0.5, 0.5, 0.5], 0.3, constraints)
```

## 🦀 Rust Usage (Library)

```rust
use fastcma::{optimize_rust, CovarianceModeKind};

let (xmin, _state) = optimize_rust(
    vec![0.5, -0.2, 0.8],
    0.3,
    None,
    Some(4000),
    Some(1e-12),
    CovarianceModeKind::Full,
    |x| x.iter().map(|v| v*v).sum()
);
println!("xmin = {:?}", xmin);
```

## 🔌 C/C++ FFI

Lightweight ABI for embedding is in `include/fastcma.h` and `include/fastcma.hpp` (version helper + sphere demo).

### Build the Shared Library

```bash
cargo build --release
```

### Use from C

```c
#include "fastcma.h"
#include <stdio.h>

int main() {
    double xmin[4];
    double f = fastcma_sphere(4, 0.4, 20000, 42ULL, xmin);
    printf("f=%g x0=%g %g %g %g\n", f, xmin[0], xmin[1], xmin[2], xmin[3]);
}
```

### Compile & Run

Compile (adjust path/extension per platform) — link against Python because the library is built with PyO3:

```bash
gcc demo.c -I include -L target/release -lfastcma $(python3-config --embed --ldflags)
```

Run (ensure the library path is visible):

```bash
LD_LIBRARY_PATH=target/release ./a.out
```

## 🌐 REST API Example

`examples/api_server.py` exposes a minimal FastAPI service wrapping `fastcma.fmin` for remote optimization jobs.

### Run the Server

```bash
uv pip install "fastapi[standard]" fast-cmaes
uvicorn examples.api_server:app --reload
```

### API Usage

POST `/optimize` with JSON body:

```json
{
  "x0": [0.5, 0.5],
  "sigma": 0.4,
  "maxfevals": 5000
}
```

Returns `xmin`, `fbest`, and evaluation counts.

## 🎨 Demos & Visualization

- 📝 `examples/python_quickstart.py` – Minimal sphere + vectorized demo
- 📊 `examples/python_benchmarks.py` – Rust vs naive Python on sphere; naive on Rastrigin
- 🎨 `examples/rich_tui_demo.py` – Rich TUI streaming sigma/fbest/evals while minimizing Rosenbrock
- 🚀 `examples/api_server.py` – FastAPI microservice for remote optimize calls

### Quick Demo Runner

One-shot setup + demo runner:

```bash
./scripts/setup_and_demo.sh
```

**What it does**: Ensures nightly Rust, creates a uv venv on Python 3.13, installs maturin + demo extras, builds the extension, and launches the Rich TUI.

### Manual TUI Setup

Run the TUI with uv + Python 3.13:

```bash
uv venv --python 3.13
uv pip install .[demo]
uv run python examples/rich_tui_demo.py
```

## 📊 Baselines & Benchmarks

- 📝 **Pure Python baseline**: `fastcma_baseline.cma_es`, `fastcma_baseline.benchmark_sphere` (see `python/fastcma_baseline/naive_cma.py`)
- ✅ **Integration benchmarks** (fixed seeds): sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, Griewank in `tests/benchmarks.rs`
- 🔥 **Hard-suite** (20 classic tough functions, seeded, higher dims): Zakharov, Levy, Dixon-Price, Powell, Styblinski–Tang, Bohachevsky, Bukin6, Dropwave, Alpine N.1, Elliptic, Salomon, Quartic, Schwefel 1.2/2.22, Bent Cigar, Rastrigin 10D, Ackley 10D, Griewank 10D, Rosenbrock 6D, Sum Squares 12D in `tests/hard_benchmarks.rs`
- 💀 **Very-hard** (ignored by default, ~20 more high-dim/fractal/ill-conditioned cases such as Katsuura, Weierstrass, Schwefel 2.26 @ 30D, HappyCat/HGBat, Expanded Schaffer F6/F7, Discus, Different Powers, 30D Rastrigin/Ackley/Rosenbrock/Griewank/Elliptic) in the same file; run with `cargo test --test hard_benchmarks -- --ignored`
- 🎨 **Rich TUI demo** for live insight

## ⚡ Performance Considerations

Performance optimization in fastcma follows a philosophy of targeted improvements rather than premature optimization. Each optimization targets a specific bottleneck identified through profiling and analysis. The result is a system that achieves substantial speedups without sacrificing code clarity or maintainability. The optimizations work synergistically: SIMD accelerates the hot path, lazy updates reduce expensive operations, and parallelization exploits modern multi-core architectures.

### Optimization Strategies

**1. SIMD Acceleration**
- **Implementation**: Uses Rust's `portable_simd` with 4 f64 lanes (256-bit AVX on x86_64)
- **Impact**: ~3-4x speedup on dot products and vector operations
- **Usage**: Automatically applied to `dot_simd()`, `square_sum_simd()`, and constraint clamping
- **When it matters**: High-dimensional problems where dot products dominate

**2. Rayon Parallelization**
- **Fitness evaluations**: `par_iter()` parallelizes objective calls across CPU cores
- **Covariance updates**: `par_chunks_mut()` parallelizes matrix operations
- **Impact**: Near-linear speedup (e.g., 8 cores → ~7x faster) on parallelizable objectives
- **Best for**: Expensive objective functions that can be evaluated independently

**3. Lazy Eigensystem Updates**
- **Strategy**: Defer expensive $$O(n^3)$$ eigen decomposition until necessary
- **Gap calculation**: $$\mathrm{lazy\_gap\_evals} = \frac{0.5 \cdot n \cdot \lambda}{(c_1 + c_{\mu}) \cdot n^2}$$
- **Impact**: Reduces eigen decompositions by 5-10x in typical runs
- **Critical for**: High-dimensional problems ($$n > 20$$) where eigen decomposition dominates

**4. Diagonal Covariance Mode**
- **When to use**: High dimensions (n > 50) or separable problems
- **Trade-off**: Faster (`O(n)` vs `O(n³)`) but assumes independent variables
- **Speedup**: 10-100x faster covariance updates, but may converge slower on correlated problems
- **Access**: Set `covariance_mode="diagonal"` in Python API

**5. Pre-allocated Buffers**
- **Buffers**: `z_buf`, `y_buf`, `tmp_buf` reused across iterations
- **Impact**: Zero-allocation inner loops, critical for low-latency applications
- **Memory**: Fixed `O(n)` overhead regardless of population size

### Robustness Features

**6. IPOP/BIPOP Restarts**
- **IPOP**: Increasing population size (`λ = base_λ * 2^restart`)
- **BIPOP**: Alternates between large (IPOP) and small populations
- **Parallel helper**: `test_utils::run_ipop_bipop_parallel` runs multiple populations concurrently
- **Python API**: `fmin_restart()` defaults to BIPOP strategy
- **Use case**: Escaping local minima without huge evaluation budgets

**7. Noise Handling**
- **Detection**: Monitors fitness spread relative to best value
- **Response**: Expands `sigma` by factor (default 1.6) when spread < threshold (default 1e-3)
- **Cooldown**: Prevents oscillation with configurable cooldown period (default 5 iterations)
- **Python API**: Enable with `noise=True` in `fmin`, `fmin_vec`, or `fmin_constrained`
- **Rust helper**: `test_utils::run_seeded_mode_noise` for deterministic testing

**8. Constraint Handling**
- **Box projection**: SIMD-accelerated clamping to `[lb, ub]` bounds
- **Mirroring**: Optional reflection of out-of-bounds points into feasible region
- **Rejection/resampling**: Up to `max_resamples` attempts if `reject()` predicate fails
- **Repair**: Optional user-provided function to fix infeasible points
- **Penalty**: Optional penalty function (includes augmented-Lagrangian helper in `test_utils`)
- **Flexibility**: Supports box constraints, nonlinear constraints, and custom feasibility rules

**9. Deterministic Seeding**
- **Implementation**: `new_with_seed()` uses `StdRng::seed_from_u64()`
- **Impact**: Same seed → identical optimization trajectory
- **Critical for**: Reproducible benchmarks, regression testing, and debugging
- **Test helpers**: `test_utils::run_seeded`, `test_utils::run_multiseed` for deterministic runs

**10. Restart Helpers**
- **Simple restarts**: `test_utils::run_with_restarts` perturbs start point between runs
- **Parallel populations**: `test_utils::run_ipop_bipop_parallel` maintains multiple concurrent populations
- **Use case**: Escaping local minima without huge evaluation budgets

## 🎛️ Feature Flags

- 📦 `numpy_support`: NumPy array support in vectorized objectives
- 🔢 `eigen_lapack`: LAPACK eigen backend
- 🧪 `test_utils`: Expose deterministic helpers externally
- 🎨 `demo`: Pulls in `rich` for the TUI

## 🛠️ Tooling & Scripts

### Setup Script (`scripts/setup_and_demo.sh`)

The setup script embodies the project's commitment to user experience. Rather than requiring users to navigate complex build requirements manually, a single command handles everything from Rust toolchain installation to running the demo. The script follows a fail-fast philosophy, checking prerequisites early and providing clear error messages when requirements aren't met.

The automation covers the complete workflow: verifying Rust nightly (required for SIMD features), installing the modern `uv` package manager, creating an isolated Python 3.13 virtual environment, installing build dependencies, compiling the Rust extension, and finally launching the Rich TUI demo. This eliminates the common friction of "getting started" that can make the project difficult to use.

**Usage**:
```bash
./scripts/setup_and_demo.sh
```

The script is idempotent and handles partial installations gracefully, making it safe to run multiple times. It's designed for users who want to explore the codebase or run a quick local demo without manual setup steps.

## 🧪 Test Suite

### Test Structure

The test suite follows a pyramid structure, with fast unit tests forming the base and comprehensive integration tests providing confidence at higher levels. This organization enables rapid feedback during development while ensuring thorough validation before releases. Each test level serves a specific purpose: unit tests catch regressions immediately, integration benchmarks validate correctness on standard problems, and hard suites stress-test robustness under extreme conditions.

The suite's design philosophy emphasizes deterministic testing through seeded random number generation. This enables reproducible test runs, making it possible to debug failures by reproducing exact optimization trajectories. The multi-seed approach validates that good performance isn't due to fortunate random initialization, while parallel test execution keeps total runtime manageable despite comprehensive coverage.

The test suite is organized into multiple levels of difficulty:

```mermaid
flowchart TD
    Start([Test Suite]) --> Core[Core Unit Tests<br/>~5-10s]
    Start --> Integration[Integration Benchmarks<br/>~10-15s]
    Start --> Hard[Hard Suite<br/>20 functions<br/>~90s]
    Start --> VeryHard[Very-Hard Suite<br/>~20 extreme cases<br/>~5-10min]
    Start --> Python[Python Integration<br/>API + Roundtrip]
    Start --> FFI[FFI Tests<br/>C/C++ bindings]
    Start --> Stability[Stability Tests<br/>Long-running]
    
    Core --> CoreTests[Sphere, Rosenbrock<br/>Noise handling<br/>Constraints<br/>Restarts]
    Integration --> IntTests[6 classic functions<br/>Multi-seed runs<br/>Parallel restarts]
    Hard --> HardTests[Zakharov, Levy, Powell<br/>Styblinski-Tang, Elliptic<br/>High-dim variants]
    VeryHard --> VHTests[Katsuura, Weierstrass<br/>30D variants<br/>Fractal functions]
    Python --> PyTests[API smoke<br/>FastAPI integration<br/>Roundtrip validation]
    FFI --> FFITests[Compile C demo<br/>Run FFI binary]
    Stability --> StabTests[Numerical stability<br/>Regression prevention]
    
    classDef coreStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef intStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef hardStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef veryHardStyle fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
    classDef otherStyle fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    
    class Core,CoreTests coreStyle
    class Integration,IntTests intStyle
    class Hard,HardTests hardStyle
    class VeryHard,VHTests veryHardStyle
    class Python,PyTests,FFI,FFITests,Stability,StabTests otherStyle
```

**Test Category Comparison:**

| Category | Functions | Runtime | Trigger | Purpose |
|----------|-----------|---------|---------|---------|
| **Core Unit** | 6 basic tests | 5-10s | Every commit | Regression detection |
| **Integration** | 6 classic functions | 10-15s | Every commit | Correctness validation |
| **Hard Suite** | 20 challenging functions | ~90s | Every commit | Stress testing |
| **Very-Hard Lite** | 3 extreme cases | ~30s | Nightly | Quick robustness check |
| **Very-Hard Full** | ~20 pathological cases | 5-10min | Weekly | Comprehensive validation |
| **Python Integration** | API + roundtrip | ~10s | Every commit | Python API correctness |
| **FFI** | C/C++ bindings | ~5s | Every commit | Native integration |

**1. Core Unit Tests** (`cargo test`)

These tests form the foundation of the test pyramid, providing rapid feedback on basic functionality. They validate core operations like sphere and Rosenbrock convergence, ensuring fundamental algorithm correctness. Additional tests cover specialized features like noise handling, constraint satisfaction through augmented Lagrangian methods, and restart mechanisms. The short runtime enables running these tests frequently during development, catching regressions before they propagate.

**2. Integration Benchmarks** (`tests/benchmarks.rs`)

Integration benchmarks validate CMA-ES correctness on six classic optimization functions that represent different problem characteristics. These tests go beyond simple convergence, exercising features like deterministic seeding, multi-seed runs that validate robustness across initializations, and parallel restart strategies. The functions span from simple convex landscapes to highly multimodal problems, ensuring the optimizer handles diverse scenarios correctly.

**3. Hard Suite** (`tests/hard_benchmarks.rs`)

The hard suite expands testing to twenty functions that stress-test the optimizer under challenging but realistic conditions. These functions introduce complications like high-order interactions, fractal landscapes, extreme condition numbers, and non-separability. Unlike unit tests that validate correctness, the hard suite validates robustness when problem characteristics deviate from ideal conditions. Functions range from low-dimensional but tricky cases like Bukin6 to high-dimensional variants of classic functions.

**4. Very-Hard Suite** (`tests/hard_benchmarks.rs -- --ignored`)

The very-hard suite represents the extreme end of testing, with approximately twenty pathological cases designed to probe algorithm limits. These include fractal functions like Katsuura and Weierstrass, high-dimensional variants up to 30D, and ill-conditioned problems that challenge numerical stability. The suite runs weekly rather than on every commit due to its computational cost, but provides essential validation that the optimizer remains robust under extreme conditions.

**5. Very-Hard Lite** (`tests/very_hard_lite.rs`)

Very-hard lite provides a compromise between thoroughness and speed, running a curated subset of three representative very-hard cases nightly. This enables regular validation of robustness without the full computational cost, catching regressions in extreme-case handling between weekly full suite runs.

**6. Python Integration Tests**
- **`tests/python_smoke.py`**: Basic Python API smoke test
- **`tests/test_api_smoke.py`**: FastAPI integration test
- **`tests/test_roundtrip.py`**: Serialization/roundtrip validation
- **`tests/python_smoke_pypi.py`**: Post-PyPI-install validation
- **Run**: `pytest tests/`

**7. FFI Tests** (`tests/ffi_smoke/`)
- **C FFI validation**: Compiles and runs C demo program
- **Purpose**: Ensure C/C++ bindings work correctly
- **Run**: `cargo build --release && gcc tests/ffi_smoke/demo.c ...`

**8. Stability Tests** (`tests/stability.rs`)
- **Long-running stability checks**: Prevents regressions
- **Purpose**: Catch numerical instabilities over many iterations

### Benchmark Functions & Their Challenges

The test suite employs a carefully curated collection of optimization functions that probe different aspects of algorithm performance. These functions span from simple convex landscapes to pathological cases with fractal structure, ensuring the optimizer handles both common scenarios and edge cases. Each function targets specific challenges: multimodality tests global search capability, ill-conditioning tests numerical stability, and non-separability tests correlation learning.

The selection follows established benchmarking practices from the optimization literature, ensuring comparability with other CMA-ES implementations and evolutionary algorithms. Functions are grouped by difficulty, with classic benchmarks providing baseline validation and harder suites stress-testing robustness under extreme conditions.

#### Classic Functions (Core Benchmarks)

These six functions form the foundation of optimization benchmarking, each exposing different algorithm characteristics. They range from the simplest convex function (Sphere) to highly multimodal landscapes (Rastrigin, Ackley) that challenge global search capabilities.

| Function | Challenge Type | Key Difficulty | Typical Dimension |
|----------|---------------|----------------|-------------------|
| **Sphere** | Convex, unimodal | Numerical precision | 6-20 |
| **Rosenbrock** | Narrow valley | Curved search path | 2-6 |
| **Rastrigin** | Highly multimodal | Many local minima | 4-10 |
| **Ackley** | Multimodal, oscillatory | Narrow global basin | 4-10 |
| **Griewank** | Non-separable | Product interactions | 5-10 |
| **Schwefel** | Deceptive | Local optima far from global | 3-10 |

**Sphere**
$$f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2$$

- **History**: Simplest convex function, introduced in optimization literature as baseline
- **Why hard**: Tests basic convergence; any optimizer should handle this
- **Challenge**: Fast convergence tests numerical precision
- **Optimum**: $$\mathbf{x}^* = \mathbf{0}$$, $$f^* = 0$$

**Rosenbrock**
$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2\right]$$

- **History**: Introduced by Howard Rosenbrock (1960) to test gradient-based methods
- **Why hard**: Narrow curved valley; gradient methods struggle with slow convergence
- **Challenge**: Tests ability to follow curved valleys (CMA-ES strength)
- **Optimum**: $$\mathbf{x}^* = (1, 1, \ldots, 1)$$, $$f^* = 0$$

**Rastrigin**
$$f(\mathbf{x}) = 10n + \sum_{i=1}^{n} \left[x_i^2 - 10\cos(2\pi x_i)\right]$$

- **History**: Named after Leonid Rastrigin (1974), designed to test global optimization
- **Why hard**: Highly multimodal with many local minima (exponentially many in high dimensions)
- **Challenge**: Tests ability to escape local minima and find global optimum
- **Optimum**: $$\mathbf{x}^* = \mathbf{0}$$, $$f^* = 0$$

**Ackley**
$$f(\mathbf{x}) = -20\exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e$$

- **History**: Proposed by David Ackley (1987) for neural network training
- **Why hard**: Many local minima with narrow global basin; exponential scaling with dimension
- **Challenge**: Tests convergence in highly multimodal landscapes
- **Optimum**: $$\mathbf{x}^* = \mathbf{0}$$, $$f^* = 0$$

**Griewank**
$$f(\mathbf{x}) = \frac{1}{4000}\sum_{i=1}^{n}x_i^2 - \prod_{i=1}^{n}\cos\left(\frac{x_i}{\sqrt{i}}\right) + 1$$

- **History**: Named after Andreas Griewank (1981), tests product term interactions
- **Why hard**: Product term creates many local minima; becomes easier in high dimensions (counterintuitive)
- **Challenge**: Tests handling of non-separable interactions
- **Optimum**: $$\mathbf{x}^* = \mathbf{0}$$, $$f^* = 0$$

**Schwefel**
$$f(\mathbf{x}) = 418.9829n - \sum_{i=1}^{n}x_i\sin(\sqrt{|x_i|})$$

- **History**: Named after Hans-Paul Schwefel, designed to test robustness
- **Why hard**: Many local minima far from global optimum; deceptive landscape
- **Challenge**: Tests ability to avoid premature convergence
- **Optimum**: $$\mathbf{x}^* \approx 420.9687$$, $$f^* \approx 0$$

#### Hard Suite Functions

The hard suite expands testing to twenty functions that stress-test the optimizer under diverse challenging conditions. These functions introduce complications like high-order interactions, fractal landscapes, extreme condition numbers, and non-separability. Unlike the classic benchmarks which primarily test convergence on well-behaved problems, the hard suite validates robustness when problem characteristics deviate from ideal conditions.

| Function | Primary Challenge | Condition Number | Separability |
|----------|------------------|------------------|--------------|
| **Zakharov** | High-order terms (quartic) | Moderate | Non-separable |
| **Levy** | Fractal-like structure | Moderate | Non-separable |
| **Dixon-Price** | Variable coupling | Moderate | Non-separable |
| **Powell** | Quartic narrow valleys | High | Non-separable |
| **Styblinski-Tang** | Multiple local minima | Moderate | Separable |
| **Bent Cigar** | Extreme ill-conditioning | $$10^6$$ | Separable |
| **Elliptic** | Exponentially increasing | $$10^{6(i-1)/(n-1)}$$ for dimension $$i$$ | Separable |
| **Schwefel 1.2** | Quadratic interactions | Moderate | Non-separable |
| **Schwefel 2.22** | Non-differentiable | Low | Separable |

**Zakharov**
$$f(\mathbf{x}) = \sum_{i=1}^{n}x_i^2 + \left(\sum_{i=1}^{n}\frac{i}{2}x_i\right)^2 + \left(\sum_{i=1}^{n}\frac{i}{2}x_i\right)^4$$

- **Challenge**: Non-separable, quartic terms create steep valleys
- **Why hard**: Tests handling of high-order interactions

**Levy**
$$f(\mathbf{x}) = \sin^2(\pi w_1) + \sum_{i=1}^{n-1}(w_i-1)^2[1+\sin^2(\pi w_{i+1})] + (w_n-1)^2[1+\sin^2(2\pi w_n)]$$

where $$w_i = 1 + \frac{x_i-1}{4}$$

- **Challenge**: Highly multimodal with many narrow valleys
- **Why hard**: Tests convergence in fractal-like landscapes

**Dixon-Price**
$$f(\mathbf{x}) = (x_1-1)^2 + \sum_{i=2}^{n}i(2x_i^2 - x_{i-1})^2$$

- **Challenge**: Non-separable, quadratic coupling between variables
- **Why hard**: Tests handling of variable dependencies

**Powell**
$$f(\mathbf{x}) = \sum_{i=1}^{n/4}\left[(x_{4i-3}+10x_{4i-2})^2 + 5(x_{4i-1}-x_{4i})^2 + (x_{4i-2}-2x_{4i-1})^4 + 10(x_{4i-3}-x_{4i})^4\right]$$

- **Challenge**: Quartic terms create narrow valleys
- **Why hard**: Tests convergence in ill-conditioned landscapes

**Styblinski-Tang**
$$f(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{n}(x_i^4 - 16x_i^2 + 5x_i)$$

- **Challenge**: Multiple local minima, non-convex
- **Why hard**: Tests global optimization capability

**Bent Cigar**
$$f(\mathbf{x}) = x_1^2 + 10^6\sum_{i=2}^{n}x_i^2$$

- **Challenge**: Extreme condition number ($$10^6$$)
- **Why hard**: Tests handling of ill-conditioned problems

**Elliptic**
$$f(\mathbf{x}) = \sum_{i=1}^{n}10^{6(i-1)/(n-1)}x_i^2$$

- **Challenge**: Exponentially increasing condition number
- **Why hard**: Tests scalability to high dimensions

**Schwefel 1.2**
$$f(\mathbf{x}) = \sum_{i=1}^{n}\left(\sum_{j=1}^{i}x_j\right)^2$$

- **Challenge**: Non-separable, quadratic interactions
- **Why hard**: Tests handling of variable coupling

**Schwefel 2.22**
$$f(\mathbf{x}) = \sum_{i=1}^{n}|x_i| + \prod_{i=1}^{n}|x_i|$$

- **Challenge**: Non-differentiable at origin, product term
- **Why hard**: Tests robustness to non-smooth landscapes

#### Very-Hard Suite Functions

**Katsuura** (`f(x) = Π[1 + (i+1)Σ|2ʲx - round(2ʲx)|/2ʲ]^(10/n^1.2) - 1`)
- **Challenge**: Fractal structure, extremely multimodal
- **Why hard**: Tests convergence in pathological landscapes

**Weierstrass** (`f(x) = ΣΣ[aᵏcos(2πbᵏ(x+0.5))] - nΣ[aᵏcos(πbᵏ)]`)
- **Challenge**: Fractal function, nowhere differentiable
- **Why hard**: Tests robustness to non-smooth landscapes

**HappyCat/HGBat**: Complex non-separable functions with multiple terms
- **Challenge**: Tests handling of intricate variable interactions

**Expanded Schaffer F6/F7**: Composed functions with oscillatory behavior
- **Challenge**: Tests convergence in oscillatory landscapes

**Discus**: Extreme condition number variant
- **Challenge**: Tests numerical stability

**Different Powers**: Variable powers create ill-conditioning
- **Challenge**: Tests handling of heterogeneous scaling

## 🔄 CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline represents a carefully orchestrated balance between thoroughness and efficiency. Rather than running all tests on every commit, the pipeline uses a tiered approach: fast tests run on every push for immediate feedback, while expensive tests run on schedules or manual triggers. This strategy keeps pull request feedback times reasonable while ensuring comprehensive validation through scheduled runs.

The pipeline's matrix strategy builds wheels for multiple operating systems and Python versions, ensuring compatibility across the supported ecosystem. Automated PyPI publishing on version tags eliminates manual deployment steps, while post-publish smoke tests validate that published packages function correctly. The inclusion of memory safety tools like Miri and address sanitizers demonstrates commitment to reliability beyond functional correctness.

The project uses GitHub Actions for comprehensive CI/CD automation. Workflow file: `.github/workflows/build-wheels.yml`

```mermaid
flowchart TD
    Trigger[Push/PR/Tag/Schedule] --> Test[test job<br/>Core tests + benchmarks]
    Trigger --> Build[build job<br/>Matrix: 3 OS × 3 Python]
    
    Test --> VeryHardLite{very-hard-lite<br/>Nightly}
    Test --> VeryHard{very-hard<br/>Weekly}
    Test --> PythonInt[integration-python<br/>Python 3.12, 3.13]
    Test --> FFIInt[integration-ffi<br/>C/C++ bindings]
    Test --> Demo[demo<br/>Setup script validation]
    Test --> Miri{miri<br/>Weekly UB detection}
    Test --> ASAN{asan<br/>Weekly memory check}
    
    Build --> BuildWheels[Build wheels<br/>Ubuntu/Windows/macOS<br/>Python 3.12/3.13/3.14]
    BuildWheels --> UploadWheels[Upload artifacts]
    
    UploadWheels --> TagCheck{Tag v*?}
    TagCheck -->|Yes| Publish[publish job<br/>Upload to PyPI]
    TagCheck -->|No| End1([End])
    
    Publish --> PyPISmoke[pypi-smoke<br/>Validate PyPI install]
    PyPISmoke --> End2([End])
    
    VeryHardLite --> End1
    VeryHard --> End1
    PythonInt --> End1
    FFIInt --> End1
    Demo --> End1
    Miri --> End1
    ASAN --> End1
    
    classDef testStyle fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef buildStyle fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef publishStyle fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    classDef decisionStyle fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    
    class Test,VeryHardLite,VeryHard,PythonInt,FFIInt,Demo,Miri,ASAN testStyle
    class Build,BuildWheels,UploadWheels buildStyle
    class Publish,PyPISmoke publishStyle
    class TagCheck decisionStyle
```

### Test Jobs

**1. `test`** (Runs on every push/PR)
- **Purpose**: Core functionality validation
- **Runs**: `cargo test --features "numpy_support test_utils"`
- **Benchmarks**: Captures timing logs for performance regression detection
- **Artifacts**: Uploads benchmark timings

**2. `very-hard-lite`** (Nightly + manual dispatch)
- **Purpose**: Quick validation of very-hard suite
- **Runs**: `cargo test --test very_hard_lite`
- **Schedule**: Daily at 5:30 AM UTC
- **Timeout**: 30 minutes

**3. `very-hard`** (Weekly + manual dispatch)
- **Purpose**: Full very-hard suite validation
- **Runs**: `cargo test --test hard_benchmarks -- --ignored`
- **Schedule**: Weekly on Sunday at 6:00 AM UTC
- **Timeout**: 60 minutes
- **Note**: Only runs on GitHub compute (expensive)

**4. `integration-python`** (Matrix: Python 3.12, 3.13)
- **Purpose**: Python API validation
- **Runs**: `pytest tests/test_api_smoke.py tests/test_roundtrip.py`
- **Tests**: FastAPI integration, roundtrip serialization

**5. `integration-ffi`**
- **Purpose**: C/C++ FFI validation
- **Builds**: `cargo build --release`
- **Compiles**: C demo program with FFI
- **Runs**: FFI smoke test binary

**6. `miri`** (Weekly + manual dispatch)
- **Purpose**: Undefined behavior detection
- **Runs**: `cargo miri test --lib`
- **Schedule**: Weekly on Sunday
- **Note**: Catches memory safety issues

**7. `asan`** (Weekly + manual dispatch)
- **Purpose**: Address sanitizer for memory errors
- **Runs**: `cargo test -Zbuild-std --target x86_64-unknown-linux-gnu --lib`
- **Schedule**: Weekly on Sunday
- **Note**: Catches use-after-free, buffer overflows

**8. `demo`**
- **Purpose**: Validates setup script and Rich TUI demo
- **Runs**: `./scripts/setup_and_demo.sh`
- **Tests**: End-to-end user experience

### Build Jobs

**9. `build`** (Matrix: 3 OS × 3 Python versions = 9 builds)
- **OS**: Ubuntu, Windows, macOS
- **Python**: 3.12, 3.13, 3.14
- **Builds**: Wheels with `maturin build --release`
- **Tests**: Installs wheel and runs Python smoke tests
- **Artifacts**: Uploads wheels for all platforms

### Publish Jobs

**10. `publish`** (Triggers on git tags `v*`)
- **Purpose**: Publish wheels to PyPI
- **Requires**: `PYPI_API_TOKEN` secret
- **Downloads**: All wheels from build artifacts
- **Publishes**: `twine upload wheels/*.whl`

**11. `pypi-smoke`** (After publish)
- **Purpose**: Validates PyPI installation
- **Installs**: `pip install fast-cmaes` from PyPI
- **Runs**: `tests/python_smoke_pypi.py`
- **Validates**: Package integrity after publish

### CI Schedule

- **Every push/PR**: Core tests, Python integration, FFI, demo, wheel builds
- **Daily (5:30 AM UTC)**: Very-hard lite suite
- **Weekly (Sunday 6:00 AM UTC)**: Full very-hard suite, Miri, ASAN

### Performance Monitoring

- **Timing logs**: Captured for benchmarks and uploaded as artifacts
- **Regression detection**: Timing comparisons across runs
- **Performance tracking**: Monitor optimization speed over time

---

## 📄 License

**MIT** (c) 2025 Jeffrey Emanuel
