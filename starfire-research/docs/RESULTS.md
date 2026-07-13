# Frozen Results Summary

## H9 — Proof-Carrying Executable Commitment State

**Terminal classification:** `PASS`

### Objective results

```text
                 stateful   endpoint   text   scalar   rewired   random-valid   invalid   delayed
training 16         16/16       0/16   0/16     0/16      0/16           0/16      0/16      0/16
holdout 8             8/8        0/8    0/8      0/8       0/8            0/8       0/8       0/8
future 32           32/32       0/32   0/32     0/32      0/32           0/32      0/32      0/32
```

### Interpretation

A validated executable commitment was necessary for later fixed-budget derivation. Preserving equivalent text, scalar information, endpoint execution, or an irrelevant/invalid/delayed mutation did not recover the effect.

### Non-claim

H9 did not infer the bridge rule; it compiled an exact witnessed relation.

---

## H10 — Evidence-Bound Rule Induction

**Terminal classification:** `PASS`

### Objective results

```text
                           training  holdout  future
stateful inferred commitment          16/16     8/8    32/32
endpoint blind                         0/16     0/8     0/32
text-only proposal                     0/16     0/8     0/32
scalar-only proposal                   0/16     0/8     0/32
rewired foreign proof                  0/16     0/8     0/32
valid permuted irrelevant inference    0/16     0/8     0/32
counterfeit proof                      0/16     0/8     0/32
delayed correct admission              0/16     0/8     0/32
```

### Frozen induction contract

```text
candidate antecedents                 = 3
candidate consequents                 = 3
candidate rules                       = 9
intervention episodes                 = 10
proposal candidate/episode evaluations = 90
validation candidate/episode evaluations = 90
executor scans                        = 3
```

### Required certificate gates

```text
score >= 10
support >= 4
contradictions == 0
winner margin >= 6
```

The validator independently recomputed every candidate score and proof field.

### Important control

The valid-permuted path inferred, validated, and admitted a structurally legitimate but objective-irrelevant rule on every root. It achieved `0/56` objective successes.

This control shows that successful proof validation and executable mutation alone do not guarantee objective relevance.

### Non-claim

H10 received a developer-supplied 3×3 candidate relation universe and a fixed scoring law.

---

## H11 — Graph-Discovered Relation Induction

**Terminal classification:** `PASS`

### Provenance

```text
preregistration commit: 3d22a43e595a547738e4964d1c57fa4fac360286
first verdict head:      667fff708d06c09b15d9b4965c53b38374c640f8
CHARGE CI run:           29143962558 (#96)
workflow conclusion:     success
artifact id:             8246133213
artifact digest:         sha256:fe5afd3070ed0789d7caece18602cce0fee60f4d10045883b07f21adbf273000
```

### Objective results

```text
                                      training  holdout  future
stateful graph-discovered inference      16/16     8/8    32/32
endpoint blind                             0/16     0/8     0/32
frontier/proof text only                   0/16     0/8     0/32
scalar only                                0/16     0/8     0/32
foreign proof                              0/16     0/8     0/32
frontier-tampered proof                    0/16     0/8     0/32
valid irrelevant graph discovery          0/16     0/8     0/32
counterfeit proof                          0/16     0/8     0/32
delayed correct admission                  0/16     0/8     0/32
```

### Discovery and compute contract

```text
raw graph atoms                          = 24
raw evidence episodes                    = 16
proposer frontier passes                 = 1
validator frontier passes                = 1
proposer graph-incidence scans           = 16
validator graph-incidence scans          = 16
discovered antecedents                   = 6
discovered consequents                   = 6
discovered candidate rules               = 36
proposal candidate/episode evaluations   = 576
validation candidate/episode evaluations = 576
admission slots                          = 1
executor scans                           = 3
independent objective checks             = 1
```

### Validation controls

```text
foreign proof rejection:             56/56
frontier-tampered proof rejection:   56/56
counterfeit proof rejection:         56/56
valid irrelevant certificate accepted: 56/56
valid irrelevant objective success:   0/56
```

### Future-family transfer

```text
cellular_regulation:       stateful 1.0, maximum control 0.0
manufacturing_processes:   stateful 1.0, maximum control 0.0
software_dependency:       stateful 1.0, maximum control 0.0
watershed_dynamics:        stateful 1.0, maximum control 0.0
```

Every root/path was executed twice from fresh state. Exact replay passed, budgets were exact, and state/provenance invariants held.

### Interpretation

H11 removed the explicit H10 candidate relation universe. The mechanism discovered a finite frontier from graph incidence, inferred a unique evidence-supported relation, and survived independent frontier rediscovery and full ranking recomputation.

### Non-claim

The representation, incidence threshold, scoring law, certificate thresholds, and finite synthetic graph construction remained developer-defined.

---

## Earlier negative results

### H4

Automatic latent-concept promotion was not justified by the observed memory-associated signal.

### H5/H5-B

Residual projection and task-profiled verification did not produce a stable, identifiable non-memory ontology target.

### H6

Resolver disagreement did not expose a sufficiently stable computational object for promotion.

### H7

State-threaded continuation did not reveal useful noncongruence beyond one-step behavior. Later resolver action was largely right-absorbing.

### H8

The experiment was classified as `CONTROL_FAILURE` because unordered internal iteration changed resolver output across nominally identical seeded runs. The apparent `NOT_COMPOSABLE` result was explicitly withdrawn as a scientific conclusion.

## Current frontier

H9-H11 establish a narrow state-transformation result. The next experimental frontier is not larger synthetic search. It is reducing developer privilege by learning reusable structural abstractions while preserving independent validation and causal controls.
