# Public Release and Sanitization Policy

The public Starfire research surface must remain separated from the private development repository and its historical data.

## Release objectives

Public releases should make the research understandable and reproducible without exposing private state or overstating the evidence.

Every release should provide:

- the frozen research question;
- preregistration and first-verdict provenance;
- mechanism description;
- exact task and compute contract;
- control definitions;
- result tables;
- replay and invariant status;
- strongest supported claim;
- explicit non-claims;
- strongest remaining limitation.

## Prohibited content

The following must never be committed to the public surface:

- SQLite databases, WAL files, or shared-memory database files;
- personal memory or identity stores;
- user conversations, private prompts, or autobiographical records;
- API keys, tokens, credentials, cookies, or environment files;
- private endpoints or infrastructure credentials;
- raw production logs containing user input;
- model weights without verified redistribution rights;
- third-party datasets without verified licenses;
- private repository history exported wholesale;
- generated artifacts that may embed sensitive state.

## Required pre-publication checks

Before code or artifacts are published:

1. inspect the exact files and generated outputs;
2. run secret scanning over the release tree;
3. search for email addresses, tokens, URLs, local paths, and database filenames;
4. confirm all datasets and dependencies have compatible licenses;
5. ensure test fixtures are synthetic;
6. verify that reports contain no private prompts or runtime memory;
7. build and test from a clean checkout;
8. compare public claims against the frozen result documents;
9. confirm that no private Git history is being exposed;
10. publish through a clean commit lineage.

## Code release standard

A sanitized reference implementation should be published only when it:

- compiles independently of the private Starfire repository;
- uses synthetic fixtures;
- contains no runtime memory integration;
- preserves deterministic operation semantics;
- includes tests for proof validation, provenance, state invariants, and replay;
- states whether it is the exact experiment implementation or a reduced reference implementation;
- includes the commit identities of any private source from which it was derived.

## Result integrity

Public result documents are append-only in meaning. Corrections are permitted, but the original result must not be silently rewritten.

A correction must state:

- what changed;
- why it changed;
- whether the terminal classification changed;
- whether code, data, thresholds, or interpretation changed;
- the identity of the superseding experiment, if applicable.

## Claim discipline

Public language must distinguish:

- direct measurement;
- architectural inference;
- research hypothesis;
- speculation.

The terms AGI, consciousness, sentience, and human-level cognition must not be used as achieved properties unless supported by a separately defined and independently validated experimental contract.

## Contribution boundary

External contributions should target the public reference implementation or new preregistered experiments. Contributions must not request or depend on private memory data, personal conversations, or production credentials.
