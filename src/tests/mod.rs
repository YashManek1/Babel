// =============================================================================
// src/tests/mod.rs  —  Operation Babel: Sprint 4.5 Physics Validation Suite
// =============================================================================
//
// MODULE LAYOUT
// -------------
//   mod.rs              ← this file: shared harness, AuditLog, TestContext
//   collision.rs        ← Layer 1a: AABB, wedge, sphere collision unit tests
//   stability.rs        ← Layer 1b: pillar jitter, sleep, stack integrity
//   stress.rs           ← Layer 1c: two-pass load propagation correctness
//   constraint.rs       ← Layer 2: runtime constraint violation checks
//   energy.rs           ← Layer 3: kinetic energy conservation and monotonicity
//
// HOW TO RUN
// ----------
//   cargo test                        # all tests, default output
//   cargo test -- --nocapture        # show println! audit logs always
//   cargo test collision              # all tests in collision module
//   cargo test test_aabb_cube_on_cube # one specific test
//
// LOGGING PHILOSOPHY
// ------------------
// Every test produces a structured AuditLog that records:
//   - what was asserted and why
//   - the observed value vs the expected value
//   - the tolerance used and whether it was met
//   - the simulation parameters (seed, block counts, step count)
//   - a PASS / FAIL verdict with a plain-English explanation
//
// Logs are always printed (via eprintln! which bypasses test capture) so they
// appear in CI regardless of --nocapture. On FAIL the log is also embedded
// in the panic message so `cargo test` shows it inline.
// =============================================================================

#![allow(dead_code)]

use std::fmt;
use std::time::Instant;

// Re-export submodules so `cargo test` discovers them
pub mod collision;
pub mod constraint;
pub mod energy;
pub mod stability;
pub mod stress_tests;

// =============================================================================
// AuditRecord — one line of structured test evidence
// =============================================================================

#[derive(Clone, Debug)]
pub struct AuditRecord {
    /// Short name of the check (e.g. "block_a_final_y")
    pub check: &'static str,
    /// Human-readable description of what is being verified
    pub description: String,
    /// The value observed from the simulation (serialised to string)
    pub observed: String,
    /// The expected value or range
    pub expected: String,
    /// Tolerance applied (ε) — "exact" if none
    pub tolerance: String,
    /// Did this check pass?
    pub passed: bool,
    /// If failed, plain-English diagnosis
    pub diagnosis: Option<String>,
}

impl fmt::Display for AuditRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "  [{verdict}] {check}\n\
                      description : {desc}\n\
                      observed    : {obs}\n\
                      expected    : {exp}\n\
                      tolerance   : {tol}",
            verdict = verdict,
            check = self.check,
            desc = self.description,
            obs = self.observed,
            exp = self.expected,
            tol = self.tolerance,
        )?;
        if let Some(diag) = &self.diagnosis {
            write!(f, "\n         diagnosis   : {}", diag)?;
        }
        Ok(())
    }
}

// =============================================================================
// SimParams — captures every parameter needed to reproduce a test run
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimParams {
    pub seed: u64,
    pub block_count: usize,
    pub steps_warmup: u32,
    pub steps_measured: u32,
    pub dt: f32,
    pub gravity_y: f32,
    pub solver_iterations: usize,
    pub floor_y: f32,
    pub extra: Vec<(String, String)>,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            seed: 42,
            block_count: 0,
            steps_warmup: 0,
            steps_measured: 0,
            dt: 1.0 / 60.0,
            gravity_y: -9.81,
            solver_iterations: 10,
            floor_y: -0.5,
            extra: vec![],
        }
    }
}

impl fmt::Display for SimParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "seed={seed}  blocks={blocks}  warmup_steps={warm}  measured_steps={meas}\n\
             dt={dt}s  gravity_y={g}  solver_iters={iters}  floor_y={floor}",
            seed = self.seed,
            blocks = self.block_count,
            warm = self.steps_warmup,
            meas = self.steps_measured,
            dt = self.dt,
            g = self.gravity_y,
            iters = self.solver_iterations,
            floor = self.floor_y,
        )?;
        for (k, v) in &self.extra {
            write!(f, "\n  {k}={v}")?;
        }
        Ok(())
    }
}

// =============================================================================
// AuditLog — collected evidence for one test case
// =============================================================================

pub struct AuditLog {
    pub test_name: String,
    pub category: &'static str,
    pub params: SimParams,
    pub records: Vec<AuditRecord>,
    pub elapsed_ms: u64,
    start: Instant,
}

impl AuditLog {
    pub fn new(test_name: impl Into<String>, category: &'static str, params: SimParams) -> Self {
        Self {
            test_name: test_name.into(),
            category,
            params,
            records: Vec::new(),
            elapsed_ms: 0,
            start: Instant::now(),
        }
    }

    // -------------------------------------------------------------------------
    // Record a generic check with full control over all fields
    // -------------------------------------------------------------------------
    pub fn record(&mut self, r: AuditRecord) {
        self.records.push(r);
    }

    // -------------------------------------------------------------------------
    // Convenience: assert two f32 values are within epsilon
    // -------------------------------------------------------------------------
    pub fn assert_approx(
        &mut self,
        check: &'static str,
        description: impl Into<String>,
        observed: f32,
        expected: f32,
        epsilon: f32,
    ) {
        let diff = (observed - expected).abs();
        let passed = diff <= epsilon;
        let diagnosis = if !passed {
            Some(format!(
                "Δ = {:.6}  exceeds ε = {:.6}.  \
                 The value drifted {:.2}× beyond tolerance. \
                 Likely cause: solver did not converge within allotted iterations, \
                 or a force/correction was applied unexpectedly.",
                diff,
                epsilon,
                diff / epsilon.max(f32::EPSILON)
            ))
        } else {
            None
        };
        self.records.push(AuditRecord {
            check,
            description: description.into(),
            observed: format!("{:.6}", observed),
            expected: format!("{:.6} ± {:.6}", expected, epsilon),
            tolerance: format!("ε = {:.6}", epsilon),
            passed,
            diagnosis,
        });
    }

    // -------------------------------------------------------------------------
    // Convenience: assert a f32 value is ≤ a threshold
    // -------------------------------------------------------------------------
    pub fn assert_leq(
        &mut self,
        check: &'static str,
        description: impl Into<String>,
        observed: f32,
        threshold: f32,
        context: impl Into<String>,
    ) {
        let passed = observed <= threshold;
        let diagnosis = if !passed {
            Some(format!(
                "Observed {:.6} exceeds threshold {:.6} by {:.6}. Context: {}",
                observed,
                threshold,
                observed - threshold,
                context.into()
            ))
        } else {
            None
        };
        self.records.push(AuditRecord {
            check,
            description: description.into(),
            observed: format!("{:.6}", observed),
            expected: format!("≤ {:.6}", threshold),
            tolerance: "strict upper bound".into(),
            passed,
            diagnosis,
        });
    }

    // -------------------------------------------------------------------------
    // Convenience: assert a boolean condition
    // -------------------------------------------------------------------------
    pub fn assert_true(
        &mut self,
        check: &'static str,
        description: impl Into<String>,
        condition: bool,
        on_fail: impl Into<String>,
    ) {
        self.records.push(AuditRecord {
            check,
            description: description.into(),
            observed: format!("{}", condition),
            expected: "true".into(),
            tolerance: "exact boolean".into(),
            passed: condition,
            diagnosis: if !condition {
                Some(on_fail.into())
            } else {
                None
            },
        });
    }

    // -------------------------------------------------------------------------
    // Convenience: assert a usize equals an expected value
    // -------------------------------------------------------------------------
    pub fn assert_count(
        &mut self,
        check: &'static str,
        description: impl Into<String>,
        observed: usize,
        expected: usize,
        on_fail: impl Into<String>,
    ) {
        let passed = observed == expected;
        self.records.push(AuditRecord {
            check,
            description: description.into(),
            observed: format!("{}", observed),
            expected: format!("{}", expected),
            tolerance: "exact integer".into(),
            passed,
            diagnosis: if !passed {
                Some(format!(
                    "Expected exactly {} but got {}. {}",
                    expected,
                    observed,
                    on_fail.into()
                ))
            } else {
                None
            },
        });
    }

    // -------------------------------------------------------------------------
    // Finalise timing and print the full log to stderr (always visible in CI)
    // -------------------------------------------------------------------------
    pub fn finish(&mut self) {
        self.elapsed_ms = self.start.elapsed().as_millis() as u64;
        eprintln!("{}", self.render());
    }

    // -------------------------------------------------------------------------
    // Did all records pass?
    // -------------------------------------------------------------------------
    pub fn all_passed(&self) -> bool {
        self.records.iter().all(|r| r.passed)
    }

    // -------------------------------------------------------------------------
    // Render the full log as a string
    // -------------------------------------------------------------------------
    pub fn render(&self) -> String {
        let total = self.records.len();
        let passed = self.records.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let overall = if failed == 0 { "PASS" } else { "FAIL" };

        let mut out = String::new();
        out.push_str(&format!(
            "\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  BABEL PHYSICS AUDIT — {overall:<39}║\n\
             ╚══════════════════════════════════════════════════════════════╝\n"
        ));
        out.push_str(&format!("  Test      : {}\n", self.test_name));
        out.push_str(&format!("  Category  : {}\n", self.category));
        out.push_str(&format!(
            "  Verdict   : {overall}  ({passed}/{total} checks passed)\n"
        ));
        out.push_str(&format!("  Duration  : {}ms\n", self.elapsed_ms));
        out.push_str("\n  ── Simulation Parameters ──────────────────────────────────\n");
        for line in self.params.to_string().lines() {
            out.push_str(&format!("  {}\n", line));
        }
        out.push_str("\n  ── Check Results ──────────────────────────────────────────\n");
        for record in &self.records {
            out.push_str(&format!("{}\n", record));
        }
        if failed > 0 {
            out.push_str(&format!(
                "\n  ✗ {failed} check(s) FAILED — see diagnosis fields above.\n"
            ));
            out.push_str(
                "  Action: fix the physics bug identified, re-run `cargo test`, \
                 confirm all checks reach PASS before proceeding to Phase II.\n",
            );
        } else {
            out.push_str(
                "\n  ✓ All checks passed. Engine behaviour is verified for this scenario.\n",
            );
        }
        out.push_str("  ───────────────────────────────────────────────────────────\n");
        out
    }

    // -------------------------------------------------------------------------
    // Call this at the end of every #[test] function.
    // Finishes timing, prints the log, and panics with the full report on fail.
    // -------------------------------------------------------------------------
    pub fn finalize_and_assert(mut self) {
        self.finish();
        if !self.all_passed() {
            panic!(
                "Physics validation FAILED — see audit log above.\n\n{}",
                self.render()
            );
        }
    }
}
