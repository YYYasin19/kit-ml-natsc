# ML Natural Sciences - Summary of Exercises

## Exercise 2

### Pandas Tutorial

- Indexing, Slicing, etc. with pandas

### Decision Trees

```python
def information_gain(df: pd.DataFrame, split_col: str, split_val: float, y_col: str)
"""
calculate information gain for cut at split_val
"""
    left_split = df[df[split_col] <= split_val]
    right_split = df[df[split_col] > split_val]
    
    e = entropy(df, 'species')
    left_e = entropy(left_split, 'species') # Entropy for left side
    right_e = entropy(right_split, 'species')
    
    # information gain = entropy - (perc_left * entropy_l) + (perc_right * entropy_r)
    IG = e - (((left_split.size/df.size)*left_e) + ((right_split.size/df.size) * right_e))
    
    return IG, left_e, right_e
```

```python
# calculates every possible split
for col in x_cols:
  values = df[col].unique()
  dfc = df[col]
  for val in values:

    # split the dataset and calculate the information gain
    IG, left_e, right_e = information_gain(df, split_col=col, split_val=val, y_col=y_col)

    # if our gain was better, replace the dictionary entry
    if IG > split_alt['ig']:
      split_alt = {
        'feature': col,
        'value': val,
        'ig': IG,
        'left_e': left_e,
        'right_e': right_e,
        'node_e':node_e
      }
      pass

    pass
```

```python
# 2 methods:

def _fit(self, df, parent=-1, level=0, direction=None)
"""Creates the tree by trying out different splits"""
```

## Exercise 3

- Calculation of vector distances using numpy

  ```python
  np.sqrt(np.sum((A[:,None] - B)**2,axis=-1))
  ```

- Implementation of precision and recall curves 

  ```python
  true_positives = np.cumsum(target[:, None] == target[resp_Sim], axis=-1)
  false_positives = np.cumsum(target[:, None] != target[resp_Sim], axis=-1) 
  total_positives = np.sum(target[:, None] == target[resp_Sim], axis=-1)
  
  # Calculate the NxN precision and recall matrices
  Pim = true_positives / (true_positives + false_positives) # of shape (1797, 1797)
  Rim = true_positives / total_positives # of shape (1797, 1797)
  ```

- Dimensionality Reduction

  - PCA (unsupervised)
  - LDA (supervised)

- Digits Dataset from sklearn

## Exercise 4

- IRIS Dataset and Linear Regression

- Implementation of Gradient Descent

  ```python
  pred = x @ omega.T
  n = x.shape[0]
  mse = pred - y
  # w' = w - a * 1/n
  omega_updated = omega - alpha * (1/n)*np.sum(mse * x, axis=0)
  
  return omega_updated
  ```

## Exercise 5

- Bayes Theorem: 
  $p(y|x) = \frac{p(x|y)p(y)}{p(x)}$​​​

## Exercise 6: Molecular Properties prediction

- LUMO and HOMO values prediction
- 65 Molecule descriptors coming from `rdkit`
- Ridge Regression: Find LUMO and HOMO values
- 70% Performance on Test set

## Exercise 7: Neural Networks for Molecular Dynamics Simulation

## Exercise 10: Graphs

- Karate Club Network
- MUTAG-Dataset for Mutagene classification
- Global Readout function of GNN is just reduce_mean
- Padding for different graph sizes
- Masking to turn zeros off

## Exercise 11: VAEs



