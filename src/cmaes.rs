use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64;

/// Output of the canonical CMA-ES optimization.
pub struct CmaesResult {
    /// Best (objective, parameters) found over the entire run (or across restarts).
    pub best_solution: (f64, Vec<f64>),
    /// Total number of generations performed (summed across all restarts).
    pub generations_used: usize,
    /// Reason for final termination.
    pub termination_reason: String,
    /// (Optional) The final population if needed.
    pub final_population: Option<Vec<(f64, Vec<f64>)>>,
}

/// Configuration for canonical CMA-ES with evolution paths and rank updates.
/// 
/// This struct contains all parameters that control the behavior of the CMA-ES algorithm.
/// Most parameters have sensible defaults and can be left as `None` to use automatic values.
/// 
/// # Basic Parameters
/// 
/// The most important parameters for typical usage are:
/// - `population_size`: Controls exploration vs exploitation trade-off
/// - `max_generations`: Maximum number of generations to run
/// - `parallel_eval`: Enable parallel function evaluation
/// - `verbosity`: Control output level
/// 
/// # Advanced Parameters
/// 
/// Advanced users can fine-tune the algorithm behavior using learning rates,
/// restart strategies, and numerical precision settings.
/// 
/// # Example
/// 
/// ```rust
/// use cmaes_lbfgs::cmaes::CmaesCanonicalConfig;
/// 
/// // Basic configuration
/// let config = CmaesCanonicalConfig {
///     population_size: 20,
///     max_generations: 1000,
///     parallel_eval: true,
///     verbosity: 1,
///     ..Default::default()
/// };
/// 
/// // Advanced configuration with restarts
/// let advanced_config = CmaesCanonicalConfig {
///     population_size: 50,
///     max_generations: 500,
///     ipop_restarts: 3,
///     total_evals_budget: 100000,
///     use_subrun_budgeting: true,
///     ..Default::default()
/// };
/// ```
pub struct CmaesCanonicalConfig {
    /// Population size (number of candidate solutions per generation).
    /// 
    /// **Default**: 0 (automatic: 4 + 3⌊ln(n)⌋ where n is problem dimension)
    /// 
    /// **Typical range**: 10-100 for most problems
    /// 
    /// **Larger values**: Better global exploration, slower convergence, more robust
    /// **Smaller values**: Faster convergence, risk of premature convergence
    /// 
    /// **Guidelines**:
    /// - Easy problems (unimodal): Use smaller populations (10-20)
    /// - Difficult problems (multimodal): Use larger populations (50-100+)
    /// - High-dimensional problems: Consider automatic sizing
    pub population_size: usize,

    /// Maximum number of generations per run.
    /// 
    /// **Default**: 500
    /// 
    /// **Guidelines**:
    /// - Simple problems: 100-500 generations usually sufficient
    /// - Complex problems: 1000-5000+ generations may be needed
    /// - Use with `total_evals_budget` for better control
    /// 
    /// If sub-run budgeting is disabled, each run uses this value for generations.
    /// Otherwise, sub-runs can get smaller or larger generation budgets.
    pub max_generations: usize,

    /// Random seed for reproducible results.
    /// 
    /// **Default**: 42
    /// 
    /// Set to different values to get different random runs, or keep the same
    /// for reproducible experiments.
    pub seed: u64,

    /// Learning rate for rank-one update of covariance matrix.
    /// 
    /// **Default**: None (automatic: 2/((n+1.3)² + μ_eff))
    /// 
    /// **Typical range**: 0.0001 - 0.1
    /// 
    /// Controls how quickly the algorithm adapts to the search direction.
    /// Smaller values = more conservative adaptation.
    pub c1: Option<f64>,

    /// Learning rate for rank-μ update of covariance matrix.
    /// 
    /// **Default**: None (automatic: depends on μ_eff and problem dimension)
    /// 
    /// **Typical range**: 0.001 - 1.0
    /// 
    /// Controls how much the population covariance influences the search distribution.
    /// Must be balanced with c1 to ensure proper covariance matrix updates.
    pub c_mu: Option<f64>,

    /// Learning rate for cumulation path for step-size control.
    /// 
    /// **Default**: None (automatic: (μ_eff + 2)/(n + μ_eff + 5))
    /// 
    /// **Typical range**: 0.1 - 1.0
    /// 
    /// Controls the step-size adaptation speed. Larger values lead to faster
    /// step-size changes but may cause instability.
    pub c_sigma: Option<f64>,

    /// Damping parameter for step-size update.
    /// 
    /// **Default**: None (automatic: 1 + 2max(0, √((μ_eff-1)/(n+1)) - 1))
    /// 
    /// **Typical range**: 1.0 - 10.0
    /// 
    /// Controls the damping of step-size updates. Larger values = more conservative
    /// step-size changes, which can improve stability but slow adaptation.
    pub d_sigma: Option<f64>,

    /// Enable parallel evaluation of candidate solutions.
    /// 
    /// **Default**: false
    /// 
    /// When true, the population is evaluated in parallel using Rayon.
    /// Recommended for expensive objective functions. Disable for very fast
    /// functions where parallelization overhead exceeds benefits.
    pub parallel_eval: bool,

    /// Verbosity level for progress output.
    /// 
    /// **Levels**:
    /// - 0: Silent (no output)
    /// - 1: Basic progress (every 10 generations)
    /// - 2: Detailed debug information
    /// 
    /// **Default**: 0
    pub verbosity: u8,

    /// Number of IPOP (Increasing Population) restarts.
    /// 
    /// **Default**: 0 (no IPOP restarts)
    /// 
    /// **Typical range**: 0-5 restarts
    /// 
    /// IPOP restarts the algorithm with increasing population sizes when
    /// it gets stuck. Each restart doubles the population size by default.
    /// Effective for multimodal problems but increases computational cost.
    /// 
    /// **Note**: BIPOP overrides IPOP if both are > 0.
    pub ipop_restarts: usize,

    /// Factor by which population size is multiplied each IPOP restart.
    /// 
    /// **Default**: 2.0
    /// 
    /// **Typical range**: 1.5 - 3.0
    /// 
    /// Controls how aggressively the population size grows with each restart.
    /// Larger factors provide better exploration but increase cost exponentially.
    pub ipop_increase_factor: f64,

    /// Number of BIPOP (Bi-Population) restarts.
    /// 
    /// **Default**: 0 (no BIPOP restarts)
    /// 
    /// **Typical range**: 0-10 restarts
    /// 
    /// BIPOP alternates between small and large population runs. More sophisticated
    /// than IPOP and often more effective for difficult multimodal problems.
    /// 
    /// **Note**: BIPOP overrides IPOP if both are > 0.
    pub bipop_restarts: usize,

    /// Total function-evaluations budget across all runs.
    /// 
    /// **Default**: 0 (no budget limit)
    /// 
    /// **Typical values**: 10,000 - 1,000,000 depending on problem complexity
    /// 
    /// When combined with `use_subrun_budgeting`, this budget is intelligently
    /// allocated across multiple restart runs.
    pub total_evals_budget: usize,

    /// Enable advanced sub-run budgeting logic for IPOP/BIPOP.
    /// 
    /// **Default**: false
    /// 
    /// When true, the `total_evals_budget` is strategically allocated across
    /// restart runs rather than running each to completion. This often provides
    /// better results within a fixed computational budget.
    pub use_subrun_budgeting: bool,

    /// Alpha parameter used in c_mu calculation.
    /// 
    /// **Default**: Some(2.0)
    /// 
    /// **Typical range**: 1.0 - 4.0
    /// 
    /// Advanced parameter that influences the balance between rank-1 and rank-μ
    /// updates of the covariance matrix. Rarely needs adjustment.
    pub alpha_mu: Option<f64>,

    /// Threshold factor for the evolution path test in step-size control.
    /// 
    /// **Default**: Some(1.4)
    /// 
    /// **Typical range**: 1.0 - 2.0
    /// 
    /// Controls when to halt the cumulation of the evolution path based on
    /// its length. Affects the balance between exploration and exploitation.
    pub hsig_threshold_factor: Option<f64>,

    /// Factor for small population size calculation in BIPOP.
    /// 
    /// **Default**: Some(0.5)
    /// 
    /// **Typical range**: 0.1 - 0.8
    /// 
    /// In BIPOP, determines the size of "small" population runs relative to
    /// the baseline population size. Smaller values = more focused local search.
    pub bipop_small_population_factor: Option<f64>,

    /// Budget allocation factor for small BIPOP runs.
    /// 
    /// **Default**: Some(1.0)
    /// 
    /// **Typical range**: 0.5 - 2.0
    /// 
    /// Controls how much of the evaluation budget is allocated to small population
    /// runs in BIPOP. Values > 1.0 give more budget to exploitation phases.
    pub bipop_small_budget_factor: Option<f64>,

    /// Budget allocation factor for large BIPOP runs.
    /// 
    /// **Default**: Some(3.0)
    /// 
    /// **Typical range**: 1.0 - 5.0
    /// 
    /// Controls how much of the evaluation budget is allocated to large population
    /// runs in BIPOP. Values > 1.0 give more budget to exploration phases.
    pub bipop_large_budget_factor: Option<f64>,

    /// Factor for large population size increase in BIPOP.
    /// 
    /// **Default**: Some(2.0)
    /// 
    /// **Typical range**: 1.5 - 3.0
    /// 
    /// Controls how the large population size grows with each BIPOP restart.
    /// Similar to `ipop_increase_factor` but for BIPOP large runs.
    pub bipop_large_pop_increase_factor: Option<f64>,

    /// Maximum number of iterations for bounds mirroring.
    /// 
    /// **Default**: Some(8)
    /// 
    /// **Typical range**: 5 - 20
    /// 
    /// When candidates violate bounds, they are "mirrored" back into the feasible
    /// region. This parameter limits how many mirror operations are performed
    /// before clamping to bounds.
    pub max_bound_iterations: Option<usize>,

    /// Numerical precision threshold for eigendecomposition convergence.
    /// 
    /// **Default**: Some(1e-15)
    /// 
    /// **Typical range**: 1e-20 to 1e-10
    /// 
    /// Controls the precision of the eigendecomposition of the covariance matrix.
    /// Smaller values = higher precision but potentially slower computation.
    pub eig_precision_threshold: Option<f64>,

    /// Minimum threshold for covariance matrix eigenvalues.
    /// 
    /// **Default**: Some(1e-15)
    /// 
    /// **Typical range**: 1e-20 to 1e-10
    /// 
    /// Prevents numerical issues by ensuring eigenvalues don't become too small.
    /// Smaller values allow more aggressive adaptation but risk numerical instability.
    pub min_eig_value: Option<f64>,

    /// Minimum threshold for matrix operations.
    /// 
    /// **Default**: Some(1e-20)
    /// 
    /// **Typical range**: 1e-25 to 1e-15
    /// 
    /// General numerical threshold for matrix computations to prevent
    /// underflow and maintain numerical stability.
    pub matrix_op_threshold: Option<f64>,

    /// Maximum stagnation generations before termination.
    /// 
    /// **Default**: Some(200)
    /// 
    /// **Typical range**: 50 - 1000
    /// 
    /// Algorithm terminates if no improvement is seen for this many consecutive
    /// generations. Prevents infinite runs on problems where the optimum has
    /// been reached within tolerance.
    pub stagnation_limit: Option<usize>,

    /// Minimum sigma value to prevent numerical issues.
    /// 
    /// **Default**: Some(1e-8)
    /// 
    /// **Typical range**: 1e-12 to 1e-6
    /// 
    /// Prevents the step-size from becoming so small that progress stops due
    /// to numerical precision limits. Should be much smaller than expected
    /// parameter scales.
    pub min_sigma: Option<f64>,
}

impl Default for CmaesCanonicalConfig {
    fn default() -> Self {
        Self {
            population_size: 0,
            max_generations: 500,
            seed: 42,
            c1: None,
            c_mu: None,
            c_sigma: None,
            d_sigma: None,
            parallel_eval: false,
            verbosity: 0,
            ipop_restarts: 0,
            ipop_increase_factor: 2.0,
            bipop_restarts: 0,
            total_evals_budget: 0,
            use_subrun_budgeting: false,
            alpha_mu: Some(2.0),
            hsig_threshold_factor: Some(1.4),
            bipop_small_population_factor: Some(0.5),
            bipop_small_budget_factor: Some(1.0),
            bipop_large_budget_factor: Some(3.0),
            bipop_large_pop_increase_factor: Some(2.0),
            max_bound_iterations: Some(8),
            eig_precision_threshold: Some(1e-15),
            min_eig_value: Some(1e-15),
            matrix_op_threshold: Some(1e-20),
            stagnation_limit: Some(200),
            min_sigma: Some(1e-8),
        }
    }
}

/// Internal struct carrying the CMA-ES "state" so we can continue from one sub-run to the next.
pub struct CmaesIntermediateState {
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub cov_eigvals: Vec<f64>,
    pub cov_eigvecs: Vec<Vec<f64>>,
    pub p_c: Vec<f64>,
    pub p_s: Vec<f64>,
    pub best_obj: f64,
    pub best_params: Vec<f64>,
}

/// Mirror reflection for bounds.
fn mirror_bound(mut x: f64, lb: f64, ub: f64, max_iterations: usize) -> f64 {
    let _width = ub - lb;
    let mut iters = 0;
    while (x < lb || x > ub) && iters < max_iterations {
        if x < lb {
            x = lb + (lb - x);
        } else if x > ub {
            x = ub - (x - ub);
        }
        iters += 1;
    }
    if x < lb {
        x = lb;
    }
    if x > ub {
        x = ub;
    }
    x
}

/// Compute CMA-ES weights.
#[allow(clippy::needless_range_loop)]
fn compute_weights(lambda: usize) -> (Vec<f64>, usize) {
    let mu = lambda / 2;
    let mut weights = vec![0.0; mu];
    let mut sum_w = 0.0;
    for i in 0..mu {
        let val = (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln();
        weights[i] = val;
        sum_w += val;
    }
    for w in weights.iter_mut() {
        *w /= sum_w;
    }
    (weights, mu)
}

/// Naive Jacobi or similar approach for a real-symmetric matrix.
#[allow(clippy::needless_range_loop)]
fn eigendecompose_symmetric(cov: &[Vec<f64>], eps: f64, apq_threshold: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let n = cov.len();
    let mut mat = cov.to_vec();
    let mut eigvecs = vec![vec![0.0; n]; n];
    for i in 0..n {
        eigvecs[i][i] = 1.0;
    }
    let mut changed = true;
    let max_iter = 64 * n;

    for _iter in 0..max_iter {
        if !changed {
            break;
        }
        changed = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[p][q];
                if apq.abs() < apq_threshold {
                    continue;
                }
                let app = mat[p][p];
                let aqq = mat[q][q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta.abs() > 1e6 {
                    0.5 / theta
                } else {
                    1.0 / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let t = if theta < 0.0 { -t } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                mat[p][p] = app - t * apq;
                mat[q][q] = aqq + t * apq;
                mat[p][q] = 0.0;
                mat[q][p] = 0.0;

                for r in 0..p {
                    let arp = mat[r][p];
                    let arq = mat[r][q];
                    mat[r][p] = arp - s * (arq + tau * arp);
                    mat[p][r] = mat[r][p];
                    mat[r][q] = arq + s * (arp - tau * arq);
                    mat[q][r] = mat[r][q];
                }
                for r in (p + 1)..q {
                    let apr = mat[p][r];
                    let arq = mat[r][q];
                    mat[p][r] = apr - s * (arq + tau * apr);
                    mat[r][p] = mat[p][r];
                    mat[r][q] = arq + s * (apr - tau * arq);
                    mat[q][r] = mat[r][q];
                }
                for r in (q + 1)..n {
                    let apr = mat[p][r];
                    let aqr = mat[q][r];
                    mat[p][r] = apr - s * (aqr + tau * apr);
                    mat[r][p] = mat[p][r];
                    mat[q][r] = aqr + s * (apr - tau * aqr);
                    mat[r][q] = mat[q][r];
                }

                for r in 0..n {
                    let vrp = eigvecs[r][p];
                    let vrq = eigvecs[r][q];
                    eigvecs[r][p] = vrp - s * (vrq + tau * vrp);
                    eigvecs[r][q] = vrq + s * (vrp - tau * vrq);
                }
                changed = true;
            }
        }
    }

    let mut diag = vec![0.0; n];
    for i in 0..n {
        diag[i] = mat[i][i];
        if diag[i] < eps {
            diag[i] = eps;
        }
    }
    Ok((eigvecs, diag))
}

#[allow(clippy::needless_range_loop)]
fn multiply_mat_vec(m: &[Vec<f64>], v: &[f64], scale: f64) -> Vec<f64> {
    let n = m.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += m[i][j] * v[j];
        }
        out[i] = scale * sum;
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn outer(v: &[f64], w: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = v[i] * w[j];
        }
    }
    mat
}

/// Multiply matrix by scalar in-place.
fn scale_mat(a: &mut [Vec<f64>], alpha: f64) {
    for row in a.iter_mut() {
        for val in row.iter_mut() {
            *val *= alpha;
        }
    }
}

/// Rebuild full covariance from B and D (diagonal).
#[allow(clippy::needless_range_loop)]
fn recompute_cov(b: &[Vec<f64>], d: &[f64]) -> Vec<Vec<f64>> {
    let n = b.len();
    let mut bd = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            bd[i][j] = b[i][j] * d[j];
        }
    }
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += bd[i][k] * b[j][k];
            }
            out[i][j] = sum;
        }
    }
    out
}

/// Inverse of B*D^0.5 for sampling or z-computation.
#[allow(clippy::needless_range_loop)]
fn invert_b_d(b: &[Vec<f64>], d: &[f64]) -> Vec<Vec<f64>> {
    let n = b.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        let inv_sd = 1.0 / (d[i].sqrt());
        for j in 0..n {
            out[j][i] = b[i][j] * inv_sd;
        }
    }
    out
}

/// Approx chi(dim).
fn chi_dim(n: f64) -> f64 {
    n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
}

/// Initializes a brand-new CMA-ES state if none is provided.
#[allow(clippy::needless_range_loop)]
fn init_cmaes_state(
    bounds: &[(f64, f64)],
    config: &CmaesCanonicalConfig,
) -> CmaesIntermediateState {
    let dim = bounds.len();
    // Mean is midpoint of bounds
    let mut mean = vec![0.0; dim];
    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        mean[i] = 0.5 * (lb + ub);
    }

    // Sigma ~ 1/4 avg range
    let mut avg_range = 0.0;
    for &(lb, ub) in bounds {
        avg_range += (ub - lb).abs();
    }
    avg_range /= dim as f64;
    let sigma = 0.25 * avg_range.max(config.min_sigma.unwrap_or(1e-8));

    // Identity for covariance
    let mut cov_eigvecs = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        cov_eigvecs[i][i] = 1.0;
    }
    let cov_eigvals = vec![1.0; dim];
    let p_c = vec![0.0; dim];
    let p_s = vec![0.0; dim];

    CmaesIntermediateState {
        mean,
        sigma,
        cov_eigvals,
        cov_eigvecs,
        p_c,
        p_s,
        best_obj: f64::INFINITY,
        best_params: vec![],
    }
}

/// Runs one CMA-ES sub-run, continuing from an existing state (if given).
/// 
/// - `state_in`: if Some, we continue from that state; else initialize fresh
/// - `best_obj_in`: the global best objective so far (to handle termination or merges)
/// - `evals_limit`: 0 => no limit, else stop if we exceed it
/// 
/// Returns:
/// - `CmaesResult` with partial info
/// - Updated state (mean, sigma, B, D, p_c, p_s, etc.)
/// - Boolean indicating if we improved the best objective in this run
/// - The number of function evaluations used in this sub-run
#[allow(clippy::too_many_arguments)]
fn cmaes_one_run<F>(
    objective: &F,
    bounds: &[(f64, f64)],
    config: &CmaesCanonicalConfig,
    state_in: Option<CmaesIntermediateState>,
    best_obj_in: f64,
    rng_seed: u64,
    evals_limit: usize,
    initial_mean: Option<&[f64]>,

) -> (CmaesResult, CmaesIntermediateState, bool, usize)
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    let dim = bounds.len();
    let mut rng = Pcg64::seed_from_u64(rng_seed);

    // CMA-ES parameters
    let lambda = if config.population_size == 0 {
        4 + (3.0 * (dim as f64).ln()) as usize
    } else {
        config.population_size
    };
    let (weights, mu) = compute_weights(lambda);
    let mu_eff = 1.0 / weights.iter().map(|w| w.powi(2)).sum::<f64>();

    // Learning rates
    let c_sigma = config
        .c_sigma
        .unwrap_or_else(|| (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0));
    let d_sigma = config
        .d_sigma
        .unwrap_or_else(|| 1.0 + 2.0_f64.max(((mu_eff - 1.0) / (dim as f64)).sqrt()));
    let c_c = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
    let c1 = config
        .c1
        .unwrap_or_else(|| 2.0 / ((dim as f64 + 1.3).powi(2) + mu_eff));
    let alpha_mu = config.alpha_mu.unwrap_or(2.0);
    let c_mu = config.c_mu.unwrap_or_else(|| {
        alpha_mu
            * ((mu_eff - 2.0 + 1.0 / mu_eff)
                / ((dim as f64 + 2.0).powi(2) + alpha_mu * mu_eff / 2.0))
    });

    // Possibly continue from existing state
    let mut st = match state_in {
        Some(s) => s,
        None => {
            let mut fresh = init_cmaes_state(bounds, config);
    
            // If user provided an initial guess, override the fresh state's mean
            if let Some(guess) = initial_mean {
                // Reflect/Clamp each dimension into [lb, ub]
                for (i, &(lb, ub)) in bounds.iter().enumerate() {
                    let g = guess[i];
                    fresh.mean[i] = mirror_bound(g, lb, ub, config.max_bound_iterations.unwrap_or(8));
                }
            }
    
            fresh
        }
    };

    // If the global best is better than the state's best, override the state's mean
    if best_obj_in < st.best_obj {
        st.best_obj = best_obj_in;
        // We do *not* forcibly override the distribution if the user wants synergy with the old cov,
        // but let's set the mean to the known best. This ensures we shift near the best region.
        st.mean = st.best_params.clone();
    }

    // If the user never updated st.best_params but best_obj_in is from outside,
    // we can forcibly store an empty state param => it won't override. So let's do an extra check:
    if st.best_params.is_empty() && best_obj_in < f64::INFINITY {
        // We don't know the param that gave best_obj_in, so we can't override the mean in that scenario.
    }

    let mut best_obj = st.best_obj;
    let mut best_params = st.best_params.clone();
    let mut mean = st.mean.clone();
    let mut sigma = st.sigma;
    let mut cov_eigvals = st.cov_eigvals.clone();
    let mut cov_eigvecs = st.cov_eigvecs.clone();
    let mut p_c = st.p_c.clone();
    let mut p_s = st.p_s.clone();

    let mut evals_used = 0usize;
    let max_generations = config.max_generations;
    let stagnation_limit = config.stagnation_limit.unwrap_or(200);

    let mut no_improvement_count = 0usize;
    let mut improved_flag = false;
    let mut generation = 0usize;
    let mut termination_reason = String::new();

    if config.verbosity >= 2 {
        eprintln!(
            "[DEBUG] Single CMA-ES run: pop_size={}, seed={}, evals_limit={}",
            lambda, rng_seed, evals_limit
        );
    }

    while generation < max_generations {
        // If we have an eval limit > 0 and used_evals >= limit, break
        if evals_limit > 0 && evals_used >= evals_limit {
            termination_reason = format!("Reached sub-run eval limit ({} evals)", evals_limit);
            break;
        }

        // Sample population
        let b = &cov_eigvecs;
        let d = &cov_eigvals;
        let candidates: Vec<Vec<f64>> = (0..lambda)
            .map(|_| {
                let mut z = vec![0.0; dim];
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim {
                    z[i] = rng.sample(StandardNormal);
                }
                let mut yz = vec![0.0; dim];
                for i in 0..dim {
                    let sqrt_d_i = d[i].sqrt();
                    let mut sum = 0.0;
                    for j in 0..dim {
                        sum += b[j][i] * z[j];
                    }
                    yz[i] = sqrt_d_i * sum;
                }
                let mut candidate = vec![0.0; dim];
                for i in 0..dim {
                    let raw = mean[i] + sigma * yz[i];
                    candidate[i] = mirror_bound(raw, bounds[i].0, bounds[i].1, config.max_bound_iterations.unwrap_or(8));
                }
                candidate
            })
            .collect();

        // Evaluate
        let evals: Vec<f64> = if config.parallel_eval {
            candidates.par_iter().map(|c| objective(c)).collect()
        } else {
            candidates.iter().map(|c| objective(c)).collect()
        };
        evals_used += candidates.len();

        if evals_limit > 0 && evals_used > evals_limit {
            termination_reason = format!(
                "Exceeded sub-run eval limit: used {}, limit={}",
                evals_used, evals_limit
            );
        }

        let mut population = Vec::with_capacity(lambda);
        for (c, fval) in candidates.into_iter().zip(evals.into_iter()) {
            population.push((fval, c));
        }
        population.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Update best
        if population[0].0 < best_obj {
            best_obj = population[0].0;
            best_params = population[0].1.clone();
            no_improvement_count = 0;
            improved_flag = true;
        } else {
            no_improvement_count += 1;
        }

        // Weighted recombination
        let mut new_mean = vec![0.0; dim];
        // mu can be up to population.len(), just in case
        let real_mu = mu.min(population.len());
        for (i, (_, p)) in population.iter().take(real_mu).enumerate() {
            let w = weights[i];
            for d_ in 0..dim {
                new_mean[d_] += w * p[d_];
            }
        }

        // Construct z_k
        let inv_bd = invert_b_d(b, d);
        let mut y_wsum = vec![0.0; dim];
        for (i, (_, p)) in population.iter().take(real_mu).enumerate() {
            let w = weights[i];
            let mut diff = vec![0.0; dim];
            for d_ in 0..dim {
                diff[d_] = p[d_] - mean[d_];
            }
            let z_k = multiply_mat_vec(&inv_bd, &diff, 1.0 / sigma);
            for d_ in 0..dim {
                y_wsum[d_] += w * z_k[d_];
            }
        }

        // Update p_s
        p_s = p_s
            .iter()
            .zip(y_wsum.iter())
            .map(|(ps_i, &y)| (1.0 - c_sigma) * ps_i + ((c_sigma * (2.0 - c_sigma) * mu_eff).sqrt()) * y)
            .collect();

        let ps_norm = p_s.iter().map(|v| v * v).sum::<f64>().sqrt();
        sigma *= f64::exp((c_sigma / d_sigma) * (ps_norm / chi_dim(dim as f64) - 1.0));

        // Update p_c
        let hsig = if ps_norm
            / (1.0 - (1.0 - c_sigma).powi(2 * (generation + 1) as i32)).sqrt()
            < (config.hsig_threshold_factor.unwrap_or(1.4) + 2.0 / (dim as f64 + 1.0)) * chi_dim(dim as f64)
        {
            1.0
        } else {
            0.0
        };
        for d_ in 0..dim {
            p_c[d_] = (1.0 - c_c) * p_c[d_] + hsig * ((c_c * (2.0 - c_c) * mu_eff).sqrt()) * y_wsum[d_];
        }

        // Covariance updates
        let mut rank1 = outer(&p_c, &p_c);
        scale_mat(&mut rank1, c1 * hsig);

        let mut rank_mu = vec![vec![0.0; dim]; dim];
        for (i, (_, p)) in population.iter().take(real_mu).enumerate() {
            let w = weights[i];
            let mut diff = vec![0.0; dim];
            for d_ in 0..dim {
                diff[d_] = p[d_] - mean[d_];
            }
            let z_k = multiply_mat_vec(&inv_bd, &diff, 1.0 / sigma);
            let o = outer(&z_k, &z_k);
            for r in 0..dim {
                for c in 0..dim {
                    rank_mu[r][c] += w * o[r][c];
                }
            }
        }
        scale_mat(&mut rank_mu, c_mu);

        let c_mat = recompute_cov(b, d);
        let mut updated_c = c_mat;
        for r in 0..dim {
            for c in 0..dim {
                updated_c[r][c] = (1.0 - c1 - c_mu) * updated_c[r][c] + rank1[r][c] + rank_mu[r][c];
            }
        }

        match eigendecompose_symmetric(&updated_c, config.eig_precision_threshold.unwrap_or(1e-15), config.matrix_op_threshold.unwrap_or(1e-20)) {
            Ok((new_b, new_d)) => {
                cov_eigvecs = new_b;
                cov_eigvals = new_d;
            }
            Err(e) => {
                if config.verbosity > 0 {
                    eprintln!("Warning: decomposition failed at gen {}: {}", generation, e);
                }
            }
        }

        mean = new_mean;

        if no_improvement_count > stagnation_limit {
            termination_reason = format!("No improvement for {} consecutive generations", stagnation_limit);
            break;
        }

        generation += 1;

        if config.verbosity > 0 && (generation % 10 == 0 || generation == max_generations) {
            println!(
                "[CMA-ES Gen {}] best={:.8}, sigma={:.5}, no_improv={}, evals_used={}",
                generation, best_obj, sigma, no_improvement_count, evals_used
            );
        }

        if !termination_reason.is_empty() {
            // e.g. we set it above for eval-limit reasons
            break;
        }
    }

    if termination_reason.is_empty() {
        termination_reason = format!("Max generations {} reached", max_generations);
    }

    // Update the state for next run
    st.mean = mean;
    st.sigma = sigma;
    st.cov_eigvals = cov_eigvals;
    st.cov_eigvecs = cov_eigvecs;
    st.p_c = p_c;
    st.p_s = p_s;
    st.best_obj = best_obj;
    st.best_params = best_params.clone();

    let result = CmaesResult {
        best_solution: (best_obj, best_params),
        generations_used: generation,
        termination_reason,
        final_population: None, // if needed, store population
    };

    (result, st, improved_flag, evals_used)
}

/// Main CMA-ES function.
pub fn canonical_cmaes_optimize<F>(
    objective: F,
    bounds: &[(f64, f64)],
    config: CmaesCanonicalConfig,
    initial_mean: Option<Vec<f64>>,

) -> CmaesResult
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    // If no restarts, just do a single run from scratch
    if config.ipop_restarts == 0 && config.bipop_restarts == 0 {
        let (res, _, _, _) = cmaes_one_run(
            &objective,
            bounds,
            &config,
            None,              // no prior state
            f64::INFINITY,     // best_obj_in
            config.seed,
            0,                 // no eval limit
            initial_mean.as_deref(), // pass the user guess as Option<&[f64]>
        );
        return res;
    }

    // BIPOP takes precedence
    if config.bipop_restarts > 0 {
        return run_bipop(&objective, bounds, config, initial_mean);
    }

    // Otherwise IPOP
    run_ipop(&objective, bounds, config, initial_mean)
}

/// IPOP with advanced state reuse.
fn run_ipop<F>(
    objective: &F,
    bounds: &[(f64, f64)],
    mut config: CmaesCanonicalConfig,
    initial_mean: Option<Vec<f64>>,

) -> CmaesResult
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    let restarts = config.ipop_restarts;
    let mut global_best = f64::INFINITY;
    let mut global_params = vec![];
    let mut final_pop = None;
    let mut final_reason = String::new();
    let mut total_gens_used = 0usize;

    // Keep track of total evals used
    let mut total_evals_used = 0usize;
    let total_budget = config.total_evals_budget;

    // Our intermediate state:
    let mut cmaes_state: Option<CmaesIntermediateState> = None;

    for restart_i in 0..=restarts {
        // Decide sub-run eval limit if sub-run budgeting is on
        let leftover = total_budget.saturating_sub(total_evals_used);
        let eval_limit_this_run = if config.use_subrun_budgeting && leftover > 0 {
            // We'll do a simple scheme: each run i gets fraction ~ (restarts - i + 1).
            // The real approach can be more refined. Here we clamp to leftover strictly.
            let runs_left = (restarts - restart_i) + 1;
            let fraction = runs_left as f64 / (restarts + 1) as f64;
            let chunk = (fraction * leftover as f64).round() as usize;
            chunk.min(leftover)
        } else {
            0 // means no limit
        };

        // Perform one run
        let (res, state_out, _improved, used_evals) = cmaes_one_run(
            objective,
            bounds,
            &config,
            cmaes_state.take(), // pass the old state to continue distribution
            global_best,
            config.seed + restart_i as u64,
            eval_limit_this_run,
            if restart_i == 0 { initial_mean.as_deref() } else { None },

        );

        // Update total usage, final state, final reason, etc.
        total_evals_used += used_evals;
        total_gens_used += res.generations_used;
        final_pop = res.final_population;
        final_reason = res.termination_reason.clone();

        // If improved, update global best
        if res.best_solution.0 < global_best {
            global_best = res.best_solution.0;
            global_params = res.best_solution.1.clone();
        }

        // Keep the new state for next iteration
        cmaes_state = Some(state_out);

        // Possibly break if we exhausted the entire budget
        if config.use_subrun_budgeting && total_budget > 0 && total_evals_used >= total_budget {
            if config.verbosity >= 1 {
                eprintln!(
                    "IPOP early stop: total_evals_used={} >= total_evals_budget={}",
                    total_evals_used, total_budget
                );
            }
            break;
        }

        // If more restarts remain, grow the population size
        if restart_i < restarts {
            config.population_size =
                (config.population_size as f64 * config.ipop_increase_factor).round() as usize;
            if config.verbosity >= 2 {
                eprintln!(
                    "[DEBUG] IPOP restart {} => new pop_size={}, total_evals_used={}",
                    restart_i + 1,
                    config.population_size,
                    total_evals_used
                );
            }
        }
    }

    CmaesResult {
        best_solution: (global_best, global_params),
        generations_used: total_gens_used,
        termination_reason: format!(
            "{} (IPOP restarts used: {})",
            final_reason, config.ipop_restarts
        ),
        final_population: final_pop,
    }
}

/// BIPOP with advanced state reuse.
fn run_bipop<F>(
    objective: &F,
    bounds: &[(f64, f64)],
    mut config: CmaesCanonicalConfig,
    initial_mean: Option<Vec<f64>>,

) -> CmaesResult
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    let restarts = config.bipop_restarts;
    let dim = bounds.len();
    let baseline_pop = if config.population_size == 0 {
        4 + (3.0 * (dim as f64).ln()) as usize
    } else {
        config.population_size
    };

    let mut global_best = f64::INFINITY;
    let mut global_params = vec![];
    let mut final_pop = None;
    let mut final_reason = String::new();
    let mut total_gens_used = 0;

    let mut total_evals_used = 0usize;
    let total_budget = config.total_evals_budget;

    // Start with no distribution (None => init on first run)
    let mut cmaes_state: Option<CmaesIntermediateState> = None;

    let mut large_pop = baseline_pop;
    let small_pop = (baseline_pop as f64 * config.bipop_small_population_factor.unwrap_or(0.5)).ceil() as usize;

    for restart_i in 0..=restarts {
        let use_small = (restart_i % 2) == 0;
        let pop_size = if use_small { small_pop } else { large_pop };

        let leftover = total_budget.saturating_sub(total_evals_used);

        // For BIPOP, let's do a different fraction for large vs. small.
        let eval_limit_this_run = if config.use_subrun_budgeting && leftover > 0 {
            let factor = if use_small { config.bipop_small_budget_factor.unwrap_or(1.0) } else { config.bipop_large_budget_factor.unwrap_or(3.0) };
            // A simple approach: fraction = factor / (restarts - restart_i + 1)
            // Then clamp.
            let runs_left = (restarts - restart_i) + 1;
            let chunk = ((factor / runs_left as f64) * leftover as f64).round() as usize;
            chunk.min(leftover)
        } else {
            0
        };

        if config.verbosity >= 2 {
            eprintln!(
                "[DEBUG] BIPOP restart {} => pop_size={}, small? {}, leftover_budget={}, evals_limit={}",
                restart_i, pop_size, use_small, leftover, eval_limit_this_run
            );
        }

        // Temporarily override config's population_size
        let old_pop = config.population_size;
        config.population_size = pop_size;

        let (res, state_out, _improved, used_evals) = cmaes_one_run(
            objective,
            bounds,
            &config,
            cmaes_state.take(),
            global_best,
            config.seed + restart_i as u64,
            eval_limit_this_run,
            if restart_i == 0 { initial_mean.as_deref() } else { None },
        );

        // Restore original pop_size in config if needed
        config.population_size = old_pop;

        total_gens_used += res.generations_used;
        total_evals_used += used_evals;
        final_pop = res.final_population;
        final_reason = res.termination_reason.clone();

        if res.best_solution.0 < global_best {
            global_best = res.best_solution.0;
            global_params = res.best_solution.1.clone();
        }

        // Save updated state for next iteration
        cmaes_state = Some(state_out);

        // If that was a large run, we double large_pop next time we do large
        // If it was a small run, we can keep small_pop or slightly adjust
        if !use_small {
            large_pop = (large_pop as f64 * config.bipop_large_pop_increase_factor.unwrap_or(2.0)).round() as usize;
        }

        if config.use_subrun_budgeting && total_budget > 0 && total_evals_used >= total_budget {
            if config.verbosity >= 1 {
                eprintln!(
                    "[DEBUG] BIPOP early stop: total_evals_used={} >= total_evals_budget={}",
                    total_evals_used, total_budget
                );
            }
            break;
        }
    }

    CmaesResult {
        best_solution: (global_best, global_params),
        generations_used: total_gens_used,
        termination_reason: format!(
            "{} (BIPOP restarts used: {})",
            final_reason, config.bipop_restarts
        ),
        final_population: final_pop,
    }
}
