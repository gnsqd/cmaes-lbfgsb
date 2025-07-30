//! Enhanced L-BFGS-B optimizer with stable dot product, adaptive finite differences,
//! and Strong Wolfe line search for robust convergence.

/// Configuration parameters for L-BFGS-B optimization.
/// 
/// This struct contains all parameters that control the behavior of the L-BFGS-B algorithm.
/// All parameters have sensible defaults suitable for most optimization problems.
/// 
/// # Core Parameters
/// 
/// The most important parameters for typical usage are:
/// - `memory_size`: Controls memory usage vs convergence speed trade-off
/// - `obj_tol` and `step_size_tol`: Control convergence criteria
/// - `c1` and `c2`: Control line search behavior
/// 
/// # Numerical Parameters
/// 
/// Parameters related to finite difference gradients and numerical stability:
/// - `fd_epsilon` and `fd_min_step`: Control gradient approximation accuracy
/// - `boundary_tol`: Controls handling of bound constraints
/// 
/// # Example
/// 
/// ```rust
/// use cmaes_lbfgs::lbfgsb_optimize::LbfgsbConfig;
/// 
/// // Basic configuration (often sufficient)
/// let config = LbfgsbConfig::default();
/// 
/// // High-precision configuration
/// let precise_config = LbfgsbConfig {
///     memory_size: 20,
///     obj_tol: 1e-12,
///     step_size_tol: 1e-12,
///     ..Default::default()
/// };
/// 
/// // Configuration for noisy functions
/// let robust_config = LbfgsbConfig {
///     c1: 1e-3,
///     c2: 0.8,
///     fd_epsilon: 1e-6,
///     max_line_search_iters: 30,
///     ..Default::default()
/// };
/// ```
pub struct LbfgsbConfig {
    /// Memory size for L-BFGS (number of past gradient vectors to store).
    /// 
    /// **Default**: 5
    /// 
    /// **Typical range**: 3-20
    /// 
    /// **Trade-offs**:
    /// - **Larger values**: Better approximation of Hessian, faster convergence, more memory
    /// - **Smaller values**: Less memory usage, more robust to non-quadratic functions
    /// 
    /// **Guidelines**:
    /// - Small problems (< 100 parameters): 5-10 is usually sufficient
    /// - Large problems (> 1000 parameters): 10-20 can help convergence
    /// - Noisy functions: Use smaller values (3-7) for more robustness
    /// - Very smooth functions: Can benefit from larger values (15-20)
    /// 
    /// **Memory usage**: Each vector stored uses O(n) memory where n is problem dimension.
    pub memory_size: usize,

    /// Tolerance for relative function improvement (convergence criterion).
    /// 
    /// **Default**: 1e-8
    /// 
    /// **Typical range**: 1e-12 to 1e-4
    /// 
    /// The algorithm terminates when the relative change in objective value falls below this threshold:
    /// `|f_old - f_new| / max(|f_old|, |f_new|, 1.0) < obj_tol`
    /// 
    /// **Guidelines**:
    /// - **High precision needed**: Use 1e-12 to 1e-10
    /// - **Standard precision**: Use 1e-8 to 1e-6
    /// - **Fast approximate solutions**: Use 1e-4 to 1e-2
    /// - **Noisy functions**: Use larger values to avoid premature termination
    pub obj_tol: f64,

    /// Tolerance for step size norm (convergence criterion).
    /// 
    /// **Default**: 1e-9
    /// 
    /// **Typical range**: 1e-12 to 1e-6
    /// 
    /// The algorithm terminates when `||step|| < step_size_tol`, indicating that
    /// parameter changes have become negligibly small.
    /// 
    /// **Guidelines**:
    /// - Should typically be smaller than `obj_tol`
    /// - For parameters with scale ~1: Use default value
    /// - For very small parameters: Scale proportionally
    /// - For very large parameters: May need to increase
    pub step_size_tol: f64,

    /// First Wolfe condition parameter (sufficient decrease, Armijo condition).
    /// 
    /// **Default**: 1e-4
    /// 
    /// **Typical range**: 1e-5 to 1e-2
    /// 
    /// Controls the required decrease in objective function for accepting a step.
    /// The condition is: `f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd`
    /// 
    /// **Trade-offs**:
    /// - **Smaller values**: More stringent decrease requirement, shorter steps, more stable
    /// - **Larger values**: Less stringent requirement, longer steps, faster progress
    /// 
    /// **Guidelines**:
    /// - **Well-conditioned problems**: Can use larger values (1e-3 to 1e-2)
    /// - **Ill-conditioned problems**: Use smaller values (1e-5 to 1e-4)
    /// - **Noisy functions**: Use smaller values for stability
    /// 
    /// **Must satisfy**: 0 < c1 < c2 < 1
    pub c1: f64,

    /// Second Wolfe condition parameter (curvature condition).
    /// 
    /// **Default**: 0.9
    /// 
    /// **Typical range**: 0.1 to 0.9
    /// 
    /// Controls the required change in gradient for accepting a step.
    /// The condition is: `|∇f(x + α*d)ᵀd| ≤ c2*|∇f(x)ᵀd|`
    /// 
    /// **Trade-offs**:
    /// - **Smaller values**: More stringent curvature requirement, shorter steps
    /// - **Larger values**: Less stringent requirement, longer steps, fewer line search iterations
    /// 
    /// **Guidelines**:
    /// - **Newton-like methods**: Use large values (0.9) to allow long steps
    /// - **Gradient descent-like**: Use smaller values (0.1-0.5) for more careful steps
    /// - **Default 0.9**: Good for L-BFGS as it allows the algorithm to take longer steps
    /// 
    /// **Must satisfy**: 0 < c1 < c2 < 1
    pub c2: f64,

    /// Base step size for finite difference gradient estimation.
    /// 
    /// **Default**: 1e-8
    /// 
    /// **Typical range**: 1e-12 to 1e-4
    /// 
    /// The actual step size used is `max(fd_epsilon * |x_i|, fd_min_step)` for each parameter.
    /// This provides relative scaling for different parameter magnitudes.
    /// 
    /// **Trade-offs**:
    /// - **Smaller values**: More accurate gradients, but risk of numerical cancellation
    /// - **Larger values**: Less accurate gradients, but more robust to noise
    /// 
    /// **Guidelines**:
    /// - **Smooth functions**: Can use smaller values (1e-10 to 1e-8)
    /// - **Noisy functions**: Use larger values (1e-6 to 1e-4)
    /// - **Mixed scales**: Ensure `fd_min_step` handles small parameters appropriately
    pub fd_epsilon: f64,

    /// Minimum step size for finite difference gradient estimation.
    /// 
    /// **Default**: 1e-12
    /// 
    /// **Typical range**: 1e-15 to 1e-8
    /// 
    /// Ensures that finite difference steps don't become too small for parameters
    /// near zero, which would lead to poor gradient estimates.
    /// 
    /// **Guidelines**:
    /// - Should be much smaller than typical parameter values
    /// - Consider the scale of your smallest meaningful parameter changes
    /// - Too small: Risk numerical precision issues
    /// - Too large: Poor gradient estimates for small parameters
    pub fd_min_step: f64,

    /// Initial step size for line search.
    /// 
    /// **Default**: 1.0
    /// 
    /// **Typical range**: 0.1 to 10.0
    /// 
    /// The line search starts with this step size and adjusts based on the Wolfe conditions.
    /// For L-BFGS, starting with 1.0 often works well as the algorithm approximates Newton steps.
    /// 
    /// **Guidelines**:
    /// - **Well-conditioned problems**: 1.0 is usually optimal
    /// - **Ill-conditioned problems**: May benefit from smaller initial steps (0.1-0.5)
    /// - **Functions with large gradients**: Consider smaller values
    /// - **Functions with small gradients**: Consider larger values
    pub initial_step: f64,

    /// Maximum number of line search iterations per optimization step.
    /// 
    /// **Default**: 20
    /// 
    /// **Typical range**: 10-50
    /// 
    /// Controls how much effort is spent finding a good step size. If the maximum
    /// is reached, the algorithm takes the best step found so far.
    /// 
    /// **Trade-offs**:
    /// - **Larger values**: More accurate line search, potentially faster overall convergence
    /// - **Smaller values**: Less time per iteration, may need more iterations overall
    /// 
    /// **Guidelines**:
    /// - **Smooth functions**: 10-20 iterations usually sufficient
    /// - **Difficult functions**: May need 30-50 iterations
    /// - **Time-critical applications**: Use smaller values (5-10)
    pub max_line_search_iters: usize,

    /// Tolerance for gradient projection to zero at boundaries.
    /// 
    /// **Default**: 1e-14
    /// 
    /// **Typical range**: 1e-16 to 1e-10
    /// 
    /// When a parameter is at a bound and the gradient would push it further beyond
    /// the bound, the gradient component is projected to zero. This tolerance
    /// determines when a parameter is considered "at" a bound.
    /// 
    /// **Guidelines**:
    /// - Should be much smaller than the expected precision of your solution
    /// - Too small: Parameters may never be considered exactly at bounds
    /// - Too large: May incorrectly project gradients for parameters near bounds
    /// - Consider the scale of your parameter bounds when setting this
    pub boundary_tol: f64,
}

impl Default for LbfgsbConfig {
    fn default() -> Self {
        Self {
            memory_size: 5,
            obj_tol: 1e-8,
            step_size_tol: 1e-9,
            c1: 1e-4,
            c2: 0.9,
            fd_epsilon: 1e-8,
            fd_min_step: 1e-12,
            initial_step: 1.0,
            max_line_search_iters: 20,
            boundary_tol: 1e-14,
        }
    }
}

/// L-BFGS-B optimizer implementation with optional configuration.
///
/// # Arguments
/// * `x` - Initial point (will be modified in-place)
/// * `bounds` - Box constraints for each parameter
/// * `objective` - Objective function to minimize
/// * `max_iterations` - Maximum number of iterations
/// * `tol` - Convergence tolerance for gradient norm
/// * `callback` - Optional callback function invoked after each iteration
/// * `config` - Optional configuration parameters (uses defaults if None)
///
/// # Returns
/// * `Result<(f64, Vec<f64>), Box<dyn std::error::Error>>` - Best objective value and parameters
pub fn lbfgsb_optimize<F, C>(
    x: &mut [f64],
    bounds: &[(f64, f64)],
    objective: &F,
    max_iterations: usize,
    tol: f64,
    callback: Option<C>,
    config: Option<LbfgsbConfig>,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>>
where
    F: Fn(&[f64]) -> f64 + Sync,
    C: Fn(&[f64], f64) + Sync,
{
    // Use provided config or default
    let config = config.unwrap_or_default();
    
    let n = x.len();
    if bounds.len() != n {
        return Err("Bounds dimension does not match x dimension.".into());
    }

    let m = config.memory_size;
    let mut s_store = vec![vec![0.0; n]; m];
    let mut y_store = vec![vec![0.0; n]; m];
    let mut rho_store = vec![0.0; m];
    let mut alpha = vec![0.0; m];

    let mut f_val = objective(x);
    let mut grad = finite_difference_gradient(x, objective, config.fd_epsilon, config.fd_min_step);
    project_gradient_bounds(x, &mut grad, bounds, config.boundary_tol);

    let mut best_f = f_val;
    let mut best_x = x.to_vec();

    let mut old_f_val = f_val;
    let mut k = 0;

    for _iteration in 0..max_iterations {
        let gnorm = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
        if gnorm < tol {
            if let Some(ref cb) = callback {
                cb(x, f_val);
            }
            return Ok((f_val, x.to_vec()));
        }

        // L-BFGS two-loop recursion
        let mut q = grad.clone();
        let nm = if k < m { k } else { m };
        for i_rev in 0..nm {
            let i_hist = (m + k - 1 - i_rev) % m;
            alpha[i_hist] = dot_stable(&s_store[i_hist], &q) * rho_store[i_hist];
            axpy(-alpha[i_hist], &y_store[i_hist], &mut q);
        }
        if nm > 0 {
            let i_last = (m + k - 1) % m;
            let sy = dot_stable(&s_store[i_last], &y_store[i_last]);
            let yy = dot_stable(&y_store[i_last], &y_store[i_last]);
            if yy.abs() > config.boundary_tol {
                let scale = sy / yy;
                for qi in q.iter_mut() {
                    *qi *= scale;
                }
            }
        }
        for i_fwd in 0..nm {
            let i_hist = (k + i_fwd) % m;
            let beta = dot_stable(&y_store[i_hist], &q) * rho_store[i_hist];
            axpy(alpha[i_hist] - beta, &s_store[i_hist], &mut q);
        }

        // Descent direction
        for d in q.iter_mut() {
            *d = -*d;
        }

        // Clamp direction
        let direction_clamped = clamp_direction(x, &q, bounds);

        // Strong Wolfe line search
        let step_size = strong_wolfe_line_search(
            x,
            f_val,
            &grad,
            &direction_clamped,
            objective,
            config.c1,
            config.c2,
            bounds,
            config.initial_step,
            config.max_line_search_iters,
            config.boundary_tol,
            config.fd_epsilon,
            config.fd_min_step,
        );

        let old_x = x.to_vec();
        for i in 0..n {
            x[i] += step_size * direction_clamped[i];
            // Enforce bounds
            if x[i] < bounds[i].0 {
                x[i] = bounds[i].0;
            } else if x[i] > bounds[i].1 {
                x[i] = bounds[i].1;
            }
        }

        // Recompute objective + gradient
        f_val = objective(x);
        grad = finite_difference_gradient(x, objective, config.fd_epsilon, config.fd_min_step);
        project_gradient_bounds(x, &mut grad, bounds, config.boundary_tol);

        // Update best solution
        if f_val < best_f {
            best_f = f_val;
            best_x = x.to_vec();
        }

        // BFGS update
        let s_vec: Vec<f64> = x.iter().zip(old_x.iter()).map(|(xi, oi)| xi - oi).collect();
        let mut y_vec = finite_difference_gradient(x, objective, config.fd_epsilon, config.fd_min_step);
        let mut old_grad = finite_difference_gradient(&old_x, objective, config.fd_epsilon, config.fd_min_step);
        project_gradient_bounds(&old_x, &mut old_grad, bounds, config.boundary_tol);
        for i in 0..n {
            y_vec[i] -= old_grad[i];
        }
        let sy = dot_stable(&s_vec, &y_vec);
        if sy.abs() > config.boundary_tol {
            let i_hist = k % m;
            s_store[i_hist] = s_vec.clone();
            y_store[i_hist] = y_vec;
            rho_store[i_hist] = 1.0 / sy;
            k += 1;
        }

        if let Some(ref cb) = callback {
            cb(x, f_val);
        }

        // Early stopping checks
        let obj_diff = (old_f_val - f_val).abs();
        if obj_diff < config.obj_tol {
            break;
        }
        old_f_val = f_val;

        let step_norm = s_vec.iter().map(|si| si * si).sum::<f64>().sqrt();
        if step_norm < config.step_size_tol {
            break;
        }
    }

    Ok((best_f, best_x))
}

/// Projects the gradient to zero at the bounds.
fn project_gradient_bounds(x: &[f64], grad: &mut [f64], bounds: &[(f64, f64)], tol: f64) {
    for i in 0..x.len() {
        let (lb, ub) = bounds[i];
        if ((x[i] - lb).abs() < tol && grad[i] > 0.0) || ((x[i] - ub).abs() < tol && grad[i] < 0.0) {
            grad[i] = 0.0;
        }
    }
}

/// Computes the finite-difference gradient using central differences with adaptive step.
fn finite_difference_gradient<F: Fn(&[f64]) -> f64>(
    x: &[f64], 
    f: &F, 
    epsilon: f64, 
    min_step: f64
) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let sqrt_epsilon = epsilon.sqrt();

    let f0 = f(x);
    for i in 0..n {
        let step = sqrt_epsilon * x[i].abs().max(epsilon);
        let step = if step < min_step { min_step } else { step };

        let mut x_forward = x.to_vec();
        x_forward[i] += step;
        let f_forward = f(&x_forward);

        let mut x_backward = x.to_vec();
        x_backward[i] -= step;
        let f_backward = f(&x_backward);

        grad[i] = (f_forward - f_backward) / (2.0 * step);
        if !f_backward.is_finite() {
            grad[i] = (f_forward - f0) / step;
        }
    }
    grad
}

/// Clamps the search direction so the step does not leave the box bounds.
fn clamp_direction(x: &[f64], d: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(d.iter())
        .zip(bounds.iter())
        .map(|((&xi, &di), &(lb, ub))| {
            if di > 0.0 {
                di.min(ub - xi)
            } else if di < 0.0 {
                di.max(lb - xi)
            } else {
                0.0
            }
        })
        .collect()
}

/// Performs y = y + alpha * x.
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Computes the dot product using Kahan summation for numerical stability.
fn dot_stable(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let product = x * y;
        let t = sum + product;
        let delta = product - (t - sum);
        c += delta;
        sum = t;
    }
    sum + c
}

/// Performs a Strong Wolfe line search and returns a suitable step size.
#[allow(clippy::too_many_arguments)]
fn strong_wolfe_line_search<F>(
    x: &[f64],
    f_val: f64,
    grad: &[f64],
    direction: &[f64],
    objective: &F,
    c1: f64,
    c2: f64,
    bounds: &[(f64, f64)],
    initial_step: f64,
    max_iter: usize,
    tol: f64,
    fd_epsilon: f64,
    fd_min_step: f64,
) -> f64
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let mut alpha_lo = 0.0;
    let mut alpha_hi = initial_step;
    let mut alpha = initial_step;

    let grad_dot_dir = dot_stable(grad, direction);

    let mut _f_lo = f_val;
    let mut _x_lo = x.to_vec();

    for _ in 0..max_iter {
        // Evaluate objective at alpha
        let trial: Vec<f64> = x.iter().zip(direction.iter())
            .enumerate()
            .map(|(idx, (&xi, &di))| {
                let candidate = xi + alpha * di;
                candidate.clamp(bounds[idx].0, bounds[idx].1)
            })
            .collect();

        let f_new = objective(&trial);

        // Armijo condition
        if f_new > f_val + c1 * alpha * grad_dot_dir {
            alpha_hi = alpha;
        } else {
            // Compute gradient at new point for curvature check
            let grad_new = finite_difference_gradient(&trial, objective, fd_epsilon, fd_min_step);
            let grad_new_dot_dir = dot_stable(&grad_new, direction);
            // Strong Wolfe second condition
            if grad_new_dot_dir.abs() <= -c2 * grad_dot_dir {
                return alpha;
            }
            if grad_new_dot_dir >= 0.0 {
                alpha_hi = alpha;
            } else {
                alpha_lo = alpha;
                _f_lo = f_new;
                _x_lo = trial;
            }
        }
        if (alpha_hi - alpha_lo).abs() < tol {
            break;
        }
        alpha = 0.5 * (alpha_lo + alpha_hi);
    }

    0.5 * (alpha_lo + alpha_hi).max(tol)
}
