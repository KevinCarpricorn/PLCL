# PLCL: Partial-Label Continual Learning

PLCL (Partial-Label Continual Learning) is a research project focusing on addressing challenges in partial-label continual learning. This project introduces an innovative weakly supervised learning algorithm that achieves performance comparable to fully supervised methods, providing a benchmark for future research in this domain.

## Overview

- **Objective**: Develop an effective weakly supervised algorithm to tackle key issues in partial-label continual learning.
- **Key Contributions**:
  - Proposed a benchmark algorithm for partial-label continual learning.
  - Achieved performance close to fully supervised methods in weakly supervised scenarios.
  - Provided a reproducible framework and comprehensive experimental results for continual learning research.

## Project Structure

- **Core Implementation**:
  - `PLCL.py`: Main algorithm implementation.
  - `model.py` & `resnet.py`: Define the model architecture and backbone network.
  - `experiment.py`: Experimental script for training and evaluation.
- **Utilities**:
  - `utils/`: Tools for loging and auxiliary algorithms.
- **Documentation**:
  - `report.pdf`: Detailed project report.

## Quick Start

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/KevinCarpricorn/PLCL.git
   cd PLCL
   ```

2.	**Run the Experiment**:
  
    ```bash
      python experiment.py
    ```
  
  Modify parameters in experiment.py as needed to customize the experimental setup.

## Results
* The proposed algorithm demonstrates robust performance in partial-label continual learning tasks.
* Experimental findings indicate comparable results to fully supervised methods in weakly supervised settings.

For detailed results and analysis, refer to report.pdf.

## License

This project is licensed under the MIT License. 
