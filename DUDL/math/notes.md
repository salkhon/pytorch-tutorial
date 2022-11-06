# Linear Algebra terminology

**Scalar**: Single object (number)

**Vector**: Column of scalars

**Matrix**: 2d space of numbers

**Tensors**: Higher dimension than matrix.

Gray scale images are stored as matrices. Colorimages are tensors.

---

Linalg data terminologies in math, np, and torch:

![linalgterms](/home/salkhon/Pictures/Screenshots/Screenshot%20from%202022-10-25%2012-21-36.png)

# Two types of reality

> Continuous:
> - Numberic
> - Many distinct values

> Categorical
> - Discrete
> - Limited distinct values

## Representing categorical data

### Dummy-coding

- 0 or 1
- Creates a single vector
- Examples: exam (pass/fail), house (sold/market), fraud detection

|Reality|y|
|---|---|
|Pass|1|
|Pass|1|
|Fail|0|

To a vector,

```python
y = [1, 1, 0]
```

### One-hot encoding

- 0 or 1 per category
- Creates a matrix
- Examples: image recognition, hand-written letter recognition

Predicting genre of movies:

Genre | History | Scifi | Kids
--- | --- | --- | ---
y1 | 0 | 1 | 0
y2 | 0 | 0 | 1
y3 | 1 | 0 | 0

To a matrix,

```python
Y = [[0, 1, 0], 
    [0, 0, 1], 
    [1, 0, 0]]
```

> One-hot encoding is a collection of multiple dummy coded features.

## Traspose

Changes row indices to column indices and vice versa

## Dot product

Element wise multiplication for vectors and matrices. 

* Matrix dot product is applied for convolution. 

Have to be the same shape. 

* A single number that reflects the commonalities between two objects (vectors, matrices, tensors, signals, images).
* In statistics, this corresponds to **correlation coefficient** or **covariance coefficient**. Infact, the correlation coefficient is nothing but the dot product between two variables. The correlational coefficient is just a fancy way of computing the dot product. 
  
The dotproduct is the computational backbone for many operations:

1. Statistics:
   1. Correlation
   2. least-squares
   3. entropy
   4. PCA
2. Signal processing:
   1. Fourier Transform
   2. filtering
3. Science:
   1. Geometry
   2. physics
   3. mechanics
4. Linear Algebra:
   1. Projection
   2. transformations
   3. multiplication
5. Deep Learning:
   1. Convolution
   2. Matrix multiplication
   3. Gram matrix (used in style transfer)

## Matrix Multiplication

Just a fancy application of the dot product. 
Dot product operands have the same dimension. So, the transpose fulfills the m x p and p x n criterion. 

## Softmax

$
\sigma_i = \frac{e^{z_i}}{\Sigma{e^z}}
$

Example:
$
z = [1, 2, 3]
$

$
e^z = [2.72, 7.39, 20.01]
$

$
\Sigma{e^z} = 30.19
$

$
\sigma = [.09, .24, .67]
$

> All of these nubmers sum to 1. That means, we can interpret these values as `probabilities`. Not probabilities of selecting the data. But since the softmax function has the property of summing to 1, we can interpret these values in terms of probabilities. 


### Why consider softmax to be probability like?
> We are going to build deep learning models that **categorize** data. They will predict category of data based on the input. The network will output a bunch of numbers for each input. There's lots of normalization, scaling, multiplication, sums and transformations, linear and non-linear steps that happen to the input on its way to the output. So the output makes no literal sense as numbers. `We take the output of the deep learning and take their softmax, and the output is a set of probabily values.`

Notice:
* Notice softmax has a non linear increase in value for the output numbers. 
* Not much of distance between small (specially negative) values. 
* Larger numbers get non linearly larger values. 

Notice:
* Adding more numbers in the softmax, lowers the softmax of individual values. 
* Sum over inputs: Any numerical value
* Sum over outputs: Guaranteed to be 1.0

![softmax](../../../../Pictures/Screenshots/Screenshot%20from%202022-10-25%2022-36-52.png)

