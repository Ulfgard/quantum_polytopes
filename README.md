# quantum_polytopes

## Requirements

We conducted the experiments with python 3.8. The following python packages must be installed:

- numpy
- scipy
- matplotlib
- cvxpy
- autograd

For the notebooks, you also need:

- plotly

## Reproducing the Plots

The raw results of our experiments are stored in a [frozen archive](https://erda.ku.dk/archives/88e993ae5f8ac761d6c2af16b6f4b953/published-archive.html). 
Please download the file and unzip it in the base directory of this repository. This will result in a folder `results/` with the results of all our scenarios:

1. ladder (S2)
2. shuttle (S1)
3. 3x3 (S3)
4. 4x4 (S4)
5. 4x4fast (S5)
6. 4x4nores (S6)

The results for each scenario and run consist of a pair of failes that contain for each run the true and estimated polytope as a python pickle and the naming convention is `truth_{scenario}_{runid}_{rho}_{invdelta}.pkl`
and  `estimate_{scenario}_{runid}_{rho}_{invdelta}.pkl`, respectively. Here, {scenario} is one of the 6 scenarios above, {runid} is a number between 1-20, rho is either 1 or 3 and represents the device difficulty and invdelta is the inverse line-search precision. 1mv is 1000 and 2mv is 500. Thus, `3x3/truth_3x3_5_3_1000.pkl` is the ground truth of the 5th run in the 3x3 scenario when rho is 3 and precision is 1mv.

To evaluate, you need to run the post processing script before running evaluate:

```
python3.8 src/postprocessing.py
python3.8 src/evaluate.py
```

This generates eps and png plots in `results/` and prints the number of successful solved tasks in Scenarios S1 and S2.



## Running the code

Single runs of the experiments can be started with
```
python3.8 src/experiment.py {scenario} {runid} {rho} {invdelta}
```

This will generate files directly in the `results/` folder. For example
```
python3.8 src/experiment.py 3x3 5 3 1000
```
Will generate the files `results/truth_3x3_5_3_1000.pkl` and `3x3/estimate_3x3_5_3_1000.pkl`.


## Tutorials

Please check the ipython Notebooks in the src repository. We advise to start with the single electron pump example before checking the 1x3 device and then the 2x2 device.