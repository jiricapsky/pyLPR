# pyLPR: python implementation of LPRules algorithm

pyLPR is implementation of LPRules algorithm from "Rule Induction in Knowledge Graphs Using Linear Programming" by Sanjeeb Dash and Joao Goncalves.

## Usage
```console
git clone https://github.com/jiricapsky/pyLPR.git
```
Install dependencies from **requirements.txt**

There are 3 main ways to run the code:
- Jupyter notebook **demo.ipynb**
- running **main.py**
- writing custom script

**main.py**

Run for *UMLS* dataset on *3 cores*, save all rules to *rules_all.npy* and save selected rules to *rules.npy*

```console
python main.py datasets/UMLS -c 3 --rules_file rules.npy --rules_file_temp rules_all.npy
```

For more information run:

```console
python main.py -h
```

**custom script**
```python
from argparse import Namespace
from pylpr.model import LPR_model

args = Namespace(
    rules_file = 'rules.npy.npy',
    rules_file_temp = 'rules_all.npy',
    solver = 'PULP_CBC_CMD',
    iterations = 20,
    rules_load = False,
    skip_writing = True,
    skip_neg = True,
    skip_weight = True,
    cores = 3,
    seed = 12345,
    max_length = 4,
    column_generation=False
)

model = LPR_model("datasets/UMLS/", [0.02, 0.03, 0.04, 0.05, 0.0055, 0.06, 0.07, 0.08, 0.09, 0.1], args)

model.fit()
result = model.predict()
```

## Testing
```console
python -m unittest discover pylpr/test
```
