# tests/

Fast CPU pytest suite. Known invariants to cover as code lands (from `docs/decisions.md`):
the D6 loss invariants (permutation of target slots leaves the loss unchanged; loss is 0 when
prediction equals target), shape contracts of the channel split (D4), and wrappers around
vendored `third_party/` code (the vendored code itself is exempt from gates; its wrappers are not).

**Owner: test-and-ci-engineer** (module authors ship smoke tests with their code).
No tests yet — this placeholder fixes ownership; `pytest` collecting 0 tests is expected.