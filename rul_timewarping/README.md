# rul_timewarping

A Python module for Remaining Useful Life (RUL) bound analysis and time warping, with bootstrap-based uncertainty quantification.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd TimeWarperRUL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Example

```python
import numpy as np
from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.bootstrap import bootstrap_inflection
from rul_timewarping.plotting import plot_g_with_inflection

# Simulate TTF data
np.random.seed(42)
ttf_data = np.random.weibull(a=2.5, size=200) * 3000

# Time warping analysis
tw = TimeWarping(ttf_data)
g_vals = tw.g_vals
t_star, _ = tw.estimate_inflection()

# Bootstrap inflection
result = bootstrap_inflection(ttf_data, B=200, alpha=0.1)

# Plot
t_grid = tw.t_grid
plot_g_with_inflection(t_grid, g_vals, t_star, ci=(result['lower'], result['upper']))
```

See `examples/demo_analysis.py` for a more detailed example.

## Module Structure
- `timewarping.py`: Core class for RUL and g(t) analysis
- `bootstrap.py`: Bootstrap logic for inflection and (future) RUL bounds
- `plotting.py`: Plotting utilities
- `utils.py`: Helper functions

## License
MIT 