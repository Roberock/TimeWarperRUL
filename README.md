![CI](https://github.com/Roberock/TimeWarperRUL/actions/workflows/python-package.yml/badge.svg)
[![Docs](https://readthedocs.org/projects/timewarperrul/badge/?version=latest)](https://timewarperrul.readthedocs.io/en/latest/)
[![Coverage](https://coveralls.io/repos/github/Roberock/TimeWarperRUL/badge.svg?branch=main)](https://coveralls.io/github/Roberock/TimeWarperRUL?branch=main)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# rul_timewarping

A Python module for Remaining Useful Life (RUL) bound analysis and time warping, with bootstrap-based uncertainty quantification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Roberock/TimeWarperRUL.git
   cd TimeWarperRUL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Example

```python

import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ttf_data = np.random.weibull(a=2.5, size=200) * 1000

    # Initialize
    tw = TimeWarping(ttf_data)

    # Access g(t) values
    g = tw.g_vals
    t = tw.t_grid


    # Get inflection points
    inflection_points_t, inflection_points_g = tw.estimate_inflection_points()

    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    plt.plot(t, g)
    plt.scatter(inflection_points_t, inflection_points_g, color='r')
    plt.grid(True)
    plt.show()
```

See `examples/demos` for a more detailed example.

## Module Structure
- `timewarping.py`: Core class for RUL and g(t) analysis
- `bootstrap.py`: Bootstrap logic for inflection and (future) RUL bounds
- `plotting.py`: Plotting utilities
- `utils.py`: Helper functions

## License
MIT 