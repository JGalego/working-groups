# Contents

- **Resahpe** operator for type [real](#real)
- **Reshape** operator for type [BOOL, STRING, BFLOAT16, FP64, FP32, FP16, INT2, INT4, INT8, INT16, INT32, INT64, UINT2, UINT4, UINT8, UINT16, UINT32, UINT64](#types)

Based on ONNX documentation version 25.

<a id="real"></a>
# **Reshape** (real)

## Signature
$Y = \text{Reshape}(X, S)$

where:
- $X$: input tensor
- $S$: desired shape on the output tensor
- $Y$: output tensor

## Restrictions
The following restrictions apply to the $\text{Reshape}$ operator for the SONNX profile:

[General Restrictions](../general_restrictions.md) are applicable

## Informal specification
Operator $\text{Reshape}$ reshapes the input tensor $X$ into an output tensor $Y$ with shape $S$. 

The reshaping is done in a way that the total number of elements remains the same, and the order of elements is preserved.

The desired shape $S$ can contain at most one dimension with the value -1. This indicates that this dimension should be inferred from the size of the input tensor and the remaining dimensions specified in $S$.

Additionally, when `allowzero` is set, any dimension in $S$ with the value 0 means that the size of that dimension should be copied from the corresponding dimension of the input tensor $X$.

In order to compute the reshape of the tensor we will divide the process into two steps:
### 1. Compute the order of the element in a flattened (1 Dimension) version of the tensor $Y$.

$$\text{flatIndex} = \sum_{i=0}^{rY-1} \left( j_i \prod_{k=i+1}^{rY-1} dY_k \right)$$

Where:
- $j_i$ is the index in dimension $i$ of tensor $Y$

### 2. Map the coordinates from the output tensor $Y$ to the input tensor $X$ using the flat index.
<a id="jx_formula"></a>

$$j_x = \left\lfloor \frac{\text{remaining}_x}{\displaystyle \prod_{k=x+1}^{rX-1} \, dX_k} \right\rfloor$$

Where:
- $x$ is the dimension index of tensor $X$

- $\text{remaining}_x = \text{flatIndex} - \displaystyle\sum_{k=0}^{x - 1} \left(j_k \cdot \displaystyle\prod_{p=k+1}^{rX-1} dX_p\right)$


Note that the product over an empty range/interval is defined to be 1 and that the sum over an empty range/interval is defined to be 0.

For instance:

- If $x = 0$, then $\text{remaining}_x$ = 0

- If $x = rX - 1$, then $\displaystyle\prod_{k=x+1}^{rX-1} dX_k$ = 1

Reshape operation can then be expressed as:

<a id="Y"></a>

$$Y[i_0, i_1, \ldots, i_{rY-1}] = X[j_0, j_1, \ldots, j_{rX-1}]$$

Where:
- $i_z \in [0, dY_z - 1]$

- $j_z \in [0, dX_z - 1]$

- $j_z$ is calculated from the [formula](#jx_formula) above.


### Example 1

```math
X = \begin{bmatrix} 
  \begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \\ 8 & 9 & 10 & 11 \end{bmatrix} 
  \begin{bmatrix} 12 & 13 & 14 & 15 \\ 16 & 17 & 18 & 19 \\ 20 & 21 & 22 & 23 \end{bmatrix}
\end{bmatrix}
```

```math
\text{S} = [1, 24]
```

```math
Y = \begin{bmatrix}\begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 \end{bmatrix}\end{bmatrix}
```
### Example 2

```math
X = \begin{bmatrix} 
  \begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \\ 8 & 9 & 10 & 11 \end{bmatrix} 
  \begin{bmatrix} 12 & 13 & 14 & 15 \\ 16 & 17 & 18 & 19 \\ 20 & 21 & 22 & 23 \end{bmatrix}
\end{bmatrix}
```

```math
\text{S} = [2, 6, 2]
```

```math
Y = \begin{bmatrix}

    \begin{bmatrix} 0 & 1 \\ 2 & 3 \\ 4 & 5 \\ 6 & 7 \\ 8 & 9 \\ 10 & 11 \end{bmatrix}
    \begin{bmatrix} 12 & 13 \\ 14 & 15 \\ 16 & 17 \\ 18 & 19 \\ 20 & 21 \\ 22 & 23 \end{bmatrix} 

    \end{bmatrix}

```
### Example 3

```math
X = \begin{bmatrix} 
  \begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \\ 8 & 9 & 10 & 11 \end{bmatrix} 
  \begin{bmatrix} 12 & 13 & 14 & 15 \\ 16 & 17 & 18 & 19 \\ 20 & 21 & 22 & 23 \end{bmatrix}
\end{bmatrix}
```

```math
\text{S} = [0, 3, -1]
```
```math
Y = \begin{bmatrix}

    \begin{bmatrix}
    0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\ 
    8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 \\ 
    16 & 17 & 18 & 19 & 20 & 21 & 22 & 23
    \end{bmatrix}

    \end{bmatrix}
```

## Error conditions
No error condition

## Inputs

### $X$: `real tensor`
Tensor $X$ is the input tensor to be reshaped.

### Constraints

 - `[C1]` <a id="C1ra"></a> Consistency between the shape of tensor $X$ the shape specified in $S$ and the shape of output tensor $Y$.

   - Statement: $prod_X = prod_{S''} = prod_Y$
    
      Where:
      - $prod_X = \displaystyle \prod_{i=0}^{rX-1}dX_i$

      - $S''$ is the restructured after $S$ (according to this [formula](#Restructure_S))

      - $prod_{S''} = \displaystyle \prod_{i=0}^{dS_0-1} S''[i]$ 

      - $prod_Y = \displaystyle \prod_{i=0}^{rY-1} dY_i$

      ### Restructured shape S
      In order to compute the reshape, we first need to determine the actual shape $S''$ of the output tensor $Y$ by handling any 0 or -1 dimensions in $S$.

      #### Handle 0 dimensions:

      <a id="Restructure_S"></a>
      $$\forall i \in [0, dS_0 - 1]:
      \quad S'[i] = \begin{cases}
        S[i] & \text{if } S[i] > 0 \lor S[i] = -1 \\[1em]
        \begin{cases}
          dX_i & \text{if } \texttt{allowzero} = 0 \\[0.3em]
          0 & \text{if } \texttt{allowzero} = 1
        \end{cases} & \text{if } S[i] = 0
      \end{cases}$$

      #### Handle -1 dimension:

      $$\forall i \in [0, dS_0 - 1]:
      \quad S''[i] = \begin{cases}
        S'[i] & \text{if } S'[i] > 0 \\[1em]
        \displaystyle\frac{prod_X}{prod_{S'}} & \text{if } S'[i] = -1 \\[1em]
      \end{cases}$$
      Where:
      - $prod_X = \displaystyle \prod_{i=0}^{rX-1} dX_i$

      - $prod_{S'} = \displaystyle \prod_{j=0, \ j \neq i}^{dS_0 - 1} S'[j]$
   
   - Rationale: Ensures that the specified shape is valid for the given tensor size.

### $S$: `int64 tensor`
Tensor $S$ specifies the desired shape of the output tensor $Y$.

### Constraints
 - `[C1]` Consistency between the shape of tensor $X$ the shape specified in $S$ and the shape of output tensor $Y$.

    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor $X$.

  - `[C2]` <a id="C2ra"></a> Value domain

    - Statement:
      
      - If `allowzero = 0`: $\forall i \in [0, dS_0 - 1] . \quad S[i] >= -1$
      
      - If `allowzero = 1`: $\forall i \in [0, dS_0 - 1]. \quad S[i] > 0 \quad \lor\\$
      
      $$(S[i] = 0 \quad \text{if} \quad \exists j \in [0, rX - 1]. \quad dX_j = 0 \quad \land \quad ( \nexists z \in [0, rX - 1]. \quad S[z] = -1) ) \quad \lor$$

      $$(S[i] = -1 \quad \text{if} \quad \nexists j \in [0, rX - 1]. \quad S[j] = 0 )$$

    - Rationale: While `allowzero` is set (equals to 1) we can only have 0 dimensions in $S$ if at least one dimension in $X$ is also 0. Also, -1 can only be used to infer a dimension if there are no zero dimensions in $X$.

  - `[C3]` <a id="C3ra"></a> Automatic shape inference

    - Statement: At most one dimension in $S$ can be -1.

    - Rationale: Ensures that the at most one dimenson can be automatically inferred.
  
  - `[C4]` <a id="C4ra"></a> Dimension size copying from input tensor $X$

    - Statement: With `allowzero` (equals to 0) not set, only valid dimensions can be copied from the input tensor $X$:
      
      - $\forall i \in [0, dS - 1]. \quad S[i] = 0 \implies i < rX$

    - Rationale: Only valid dimensions in $X$ can be copied to the output.
 
## Attributes

### allowzero: `integer`
If set to 0, dimensions in $S$ with value 0 will copy the size from the corresponding dimension of the input tensor $X$. 

If set to 1, any 0 in $S$ means that the size of that dimension should be 0.
In this context -1 can only be used to infer a dimension if there are no zero dimensions in $S$.

  - `[C1]` <a id="C2ra"></a> Value domain

    - Statement: `allowzero` only takes values 0 or 1.

 - `[C2]` <a id="C3ra"></a> Value domain in $S$

    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C2]</span></b>](#C2ra) on tensor $S$.
  
  - `[C4]` <a id="C4ra"></a> Shape copying from input tensor $X$

    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C4]</span></b>](#C4ra) on tensor $S$.

    - Rationale: Only valid dimensions in $X$ can be copied to the output.

## Outputs

### $Y$: `real tensor`
Tensor $Y$ is the reshaped output tensor.

### Constraints

 - `[C1]` Shape consistency
   - Statement: See constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor $X$.
   
## Formal specification
See the Why3 specification.

## Numerical Accuracy
The $\text{Reshape}$ operator does not introduce any numerical error. Hence, for all valid indices the output values are exactly equal to the corresponding input values.