# Sceptical LLMs

**When the problem is well-posed, reason correctly. When it is not, say so.**

Frontier language models are fluent at applying Bayes’ rule to vignettes. The harder question is whether they are *sceptical*: do they notice patently false premises, inconsistent priors, overlapping pathways left unspecified, and traps dressed up as word problems—or do they compute anyway and report a number?

This project benchmarks **meta-reasoning under uncertainty**: the discipline of knowing when a posterior is warranted, and when the honest answer is *insufficient information* or *the setup is inconsistent*.

---

## Manifesto

A sceptical model knows the normative answer when the graph is fully specified:

$$
P(A \mid T) =
\frac{(q_c s_c + q_d s_d)\,P(A)}
{(q_c s_c + q_d s_d)\,P(A) + r_e s_e\,P(B) + f_n\,P(N)}
$$

**Notation** (multi-cause base-rate graph):

| Symbol | Meaning |
|--------|---------|
| \(A, B, N\) | Mutually exclusive causes; \(P(N) = 1 - P(A) - P(B)\) |
| \(C, D\) | Conditions driven by \(A\); mutually exclusive given \(A\) |
| \(E\) | Condition driven by \(B\) |
| \(T\) | Positive test |
| \(q_c, q_d\) | \(P(C \mid A)\), \(P(D \mid A)\); require \(q_c + q_d \le 1\) |
| \(r_e\) | \(P(E \mid B)\) |
| \(s_c, s_d, s_e\) | \(P(T \mid C)\), \(P(T \mid D)\), \(P(T \mid E)\) |
| \(f_n\) | \(P(T \mid N)\) — baseline false-positive rate |

Likelihoods:

$$
P(T \mid A) = q_c s_c + q_d s_d, \qquad
P(T \mid B) = r_e s_e, \qquad
P(T \mid N) = f_n
$$

The manifesto is not the formula alone. It is the **pair**:

1. **Compute** \(P(A \mid T)\) when the stated probabilities define a coherent model.
2. **Refuse** when they do not—when priors do not partition, pathways overlap without specification, or opening claims contradict known facts and the problem never reconciles them.

Blind application of the numerator is not intelligence; it is pattern matching. Scepticism is the guardrail.

---

## What we test

| Failure mode | Sketch |
|--------------|--------|
| **False premises** | Real-world claims (prevalence, epidemiology) that are patently wrong |
| **Inconsistent priors** | \(P(A) + P(B) + P(N) \neq 1\) |
| **Inconsistent pathways** | \(q_c + q_d > 1\) or other structural violations |
| **Underdetermined overlap** | \(C\) and \(D\) not stated mutually exclusive |
| **Lure shortcuts** | \(q_c s_c + q_d s_d\) or \(P(T \mid A)\) mistaken for \(P(A \mid T)\) |

Items are drawn from domains where humans and models alike reach for a number—medicine, law, security, marketing—and vary only in whether the setup *earns* that computation.

---

## Status

Early bootstrap. Benchmark items, scoring, and Kaggle tasks will land here as the suite is built.

Sibling work: [cognitive-biases-in-llms](../cognitive-biases-in-llms) (bias susceptibility on Kaggle Community Benchmarks).

---

## License

TBD.
