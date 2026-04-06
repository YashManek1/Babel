# BABEL — Physics Validation Log

## Sprint 4.5: Research Integrity Layer

---

## March 30 — Validation Run

### Summary

- Total Tests: 32
- Passed: 19
- Failed: 13
- Status: FAILED

---

## Failure Analysis

---

### 1. Collision — Heavy vs Light Mass Sharing

**Test:** `test_aabb_heavy_vs_light_mass_sharing`
**Category:** Collision
**Status:** Failed

#### Observations

- Heavy block displacement is correct
- Heavy block velocity becomes zero (incorrect)

#### Diagnosis

- Heavy velocity after collision: `0.0`
- Expected: greater than `0.5 m/s`

#### Root Cause

- Velocity update is missing or incorrect after positional correction
- XPBD correction is applied, but momentum transfer is not preserved

#### Fix Plan

- Apply velocity update after position correction:

```rust
v += (x_new - x_old) / dt;
```

- Verify impulse distribution based on inverse mass

#### Expected Outcome

- Heavy block continues forward with minimal velocity loss
- Light block absorbs the majority of impulse

---

### 2. Collision — Equal Mass Transfer

**Test:** `test_aabb_two_dynamic_equal_mass_collision`
**Category:** Collision
**Status:** Failed

#### Observations

- Momentum conservation is correct
- Energy conservation is correct
- Block B does not acquire velocity

#### Diagnosis

- Block B velocity after collision: `0.0`

#### Root Cause

- Collision response is not symmetric
- Constraint correction applied to only one body or incorrect normal direction

#### Fix Plan

- Apply symmetric correction:

```rust
delta_a = -normal * correction * inv_mass_a / (inv_mass_a + inv_mass_b);
delta_b =  normal * correction * inv_mass_b / (inv_mass_a + inv_mass_b);
```

#### Expected Outcome

- Block A slows down
- Block B gains positive velocity

---

### 3. Collision — Sleep System Failure

**Test:** `test_aabb_cube_on_static_floor`
**Category:** Collision
**Status:** Failed

#### Observations

- Position is correct
- Velocity is zero
- Sleep state not triggered

#### Diagnosis

- `is_sleeping = false` after 300 steps

#### Root Cause

- Sleep threshold logic not applied or not triggered for floor contacts

#### Fix Plan

```rust
if velocity.length() < LINEAR_SLEEP_SPEED {
    entity.sleeping = true;
}
```

#### Expected Outcome

- Block transitions to sleeping state after settling

---

### 4. Constraint — Penetration Violation

**Test:** `test_constraint_no_penetration_100_steps`
**Category:** Constraint
**Status:** Failed

#### Observations

- Maximum penetration: `0.966691`
- Allowed maximum: `0.01`

#### Diagnosis

- Constraint violation occurs at step 36
- Solver fails to converge

#### Root Cause

- Insufficient solver iterations or incorrect constraint formulation
- XPBD lambda/compliance update may be incorrect

#### Fix Plan

- Increase solver iterations (e.g., 10 → 20+)
- Validate constraint formulation:

```rust
C(x) = distance - rest_length
lambda += -C / (w_sum + compliance)
```

#### Expected Outcome

- Maximum penetration remains below `0.01` across all steps

---

## Passing Systems

The following systems are verified as correct:

- Static body immobility
- Sphere-floor collision
- Energy dissipation
- Cube stacking
- Wedge sliding behavior
- Adhesion constraints

These indicate that:

- Broad-phase collision detection is functioning correctly
- Core solver is partially correct
- Issues are localized rather than systemic

---

## Key Metrics

| Metric             | Value    | Status |
| ------------------ | -------- | ------ |
| Max Penetration    | 0.966691 | Failed |
| Momentum Creation  | 0        | Passed |
| Energy Creation    | 0        | Passed |
| Sleep Activation   | Failed   | Failed |
| Collision Transfer | Partial  | Failed |

---

## Global Root Causes

1. Missing velocity updates after positional correction
2. Asymmetric collision handling
3. Weak constraint solver convergence
4. Sleep system not properly integrated

---

## Action Plan

### Step 1 (Critical)

Fix velocity update after XPBD correction

### Step 2

Fix symmetric collision response

### Step 3

Improve constraint solver convergence

- Increase iterations
- Validate lambda update

### Step 4

Fix sleep system triggering

---

## Definition of Done

Before progressing to Sprint 5:

- [ ] Maximum penetration < 0.01
- [ ] Correct momentum transfer
- [ ] No velocity loss issues
- [ ] Sleep system functioning correctly
- [ ] All tests passing

---

## Notes

This document serves as the authoritative validation record for the physics engine.
Future validation runs should be appended with new dated entries rather than replacing existing data.
