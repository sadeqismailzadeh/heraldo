# Specifying Measurement Patterns (Heralding Outcomes)

When running an optimization or performing post‑hoc analysis, you often need to provide a set of **fixed photon‑number patterns** that define which ancillary measurement outcomes are accepted. The `measurement_patterns` argument to `BasinHoppingRunner` and the `outcomes` argument to analysis functions (`analyze_rotations`, `analyze_loss`, `analyze_cutoff`, `plot_outcomes`) accept a wide variety of input formats.

The underlying normalisation is handled by `heraldo._internal._normalize_outcomes`, which converts all inputs into a **list of tuples of integers**, where each tuple has length equal to the number of measured ancillary modes.

---

## Number of Measured Modes

- **2‑mode circuits** (e.g., `TwoModeStaticGeneral`, `TwoModeStaticSqueezeOnly`):  
  Only **Mode 1** is measured.  
  Each pattern is a single integer → `(n1,)`.

- **3‑mode circuits** (e.g., `ThreeModeStaticGeneral`, `ThreeModeStaticSqueezeOnly`):  
  Modes **1 and 2** are measured.  
  Each pattern is a pair of integers → `(n1, n2)`.

---

## Accepted Input Formats

`measurement_patterns` (and `outcomes`) can be:

| Format                         | Example (2‑mode)                | Example (3‑mode)                  | Resulting tuples         |
|--------------------------------|--------------------------------|----------------------------------|---------------------------|
| **`None`**                     | `None`                         | `None`                           | Beam search (no fixed)    |
| **Single integer**             | `4`                            | ❌ Not allowed (would fail)      | `[(4,)]`                  |
| **Tuple of integers**          | `(4,)`                         | `(2, 4)`                         | `[(4,)]` or `[(2, 4)]`    |
| **List of integers**           | `[4, 5, 6]`                    | ❌ Not allowed (would fail)      | `[(4,), (5,), (6,)]`      |
| **List of lists**              | `[[4], [5]]`                   | `[[2, 4], [4, 2]]`               | `[(4,), (5,)]` or `[(2, 4), (4, 2)]` |
| **List of tuples**             | `[(4,), (5,)]`                 | `[(2, 4), (4, 2)]`               | `[(4,), (5,)]` or `[(2, 4), (4, 2)]` |
| **Numpy array** (any shape)    | `np.array([4, 5])`             | `np.array([[2,4], [4,2]])`       | `[(4,), (5,)]` or `[(2, 4), (4, 2)]` |
| **Empty list**                 | `[]`                           | `[]`                             | No patterns |

> **Important**: For 2‑mode circuits, any format that yields a flat list of integers is expanded to a list of length‑1 tuples.  
> For 3‑mode circuits, you **must** provide tuples or lists of length 2; passing a flat list of integers will raise a `ValueError`.


**See also**:  
- `heraldo._internal._normalize_outcomes` source code.  