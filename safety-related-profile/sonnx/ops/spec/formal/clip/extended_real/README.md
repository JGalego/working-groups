# How to represent Nans in Real domain

## Problem Statement
On the concrete formalization (C formalization), it its possible to have values like Nan, Inf, -Inf, the problem is how to map the results of C operation to a Real domain operation.


## Proposed Solution
### Solution 1: Use float why3 type - Not recommended
Instead of usin Real type on the abstract formalization, we can use float type which already has the Nan, Inf, -Inf values. The downside of this solution is that we will, basically, lose the abstract layer. This solution will be very close to the concrete formalization.

### Solution 2: Use a middle abstract layer
We can define a new type in the abstract formalization to represent the float values, *extended_real*, which includes the special values Nan, Inf, and -Inf, along with the regular real numbers. This allows us to maintain an abstract layer while still handling these special cases explicitly.

The problem now is on the abstract side, we need to map values from extended_real to Real.

#### Solution 2.1: Use constant on Real domain
We can define three constants in the Real domain to represent Nan, Inf, and -Inf and redefine the operations accordingly. For example:
```why3
    let ghost function sum (r1 r2: real) : real =
        if is_nan r1 || is_nan r2 then
            not_a_number
        else if is_infinite r1 && is_infinite r2 then
            inf_real
        else if is_neg_infinite r1 && is_neg_infinite r2 then
            neg_inf_real
        else if (is_infinite r1 && is_neg_infinite r2) || (is_neg_infinite r1 && is_infinite r2) then
            not_a_number
        else if is_infinite r1 then
            inf_real
        else if is_infinite r2 then
            inf_real
        else if is_neg_infinite r1 then
            neg_inf_real
        else if is_neg_infinite r2 then
            neg_inf_real
        else
            r1 + r2
```
With this approach, we can handle the special cases while still using the Real type for regular numbers. But now we need to define the properties of these constants and how they interact with regular real numbers. This approach can be seen on the file `extended_real.mlw`.
The downside of this approach is that it is close to the extended_real, because of the use of constants to represent special values.

#### Solution 2.2: Use predicates to identify special values
Instead of using constants to represent Nan, Inf, and -Inf, we can just declare a predicate to represent each special value. For example:
```why3
    (* Checks if a real is positive infinite *)
    predicate is_infinite (r:real)
```
Here we try 2 different approaches:
1. Just define the operations on extended_real, using val functions.
2. Define the operations on Real using the predicates to identify special values and define the operation on extended_real, using val functions.

For both approaches, we create a very simple operator, to show how it works.

On the file `extended_real_no_constants.mlw`, you can see the first approach if you check the ghost function `extended_real_add_operator`, particularly this ensures
```why3
ensures { ext_tensor_to_tensor_real result =
                    add_operator (ext_tensor_to_tensor_real t1) (ext_tensor_to_tensor_real t2) }
```
You will see that we cannot prove this property because, on our opinion, we don't have enough information about the behavior of the add_operator when one of the inputs is a special value.


On the file `extended_real_no_constants_2.mlw`, you can see the second approach, on the ghost function `add_ext_operation`, its now possible to prove the refinement mapping.
```why3
        ensures { (ext_tensor_to_tensor_real result) = add_operator (ext_tensor_to_tensor_real t1) (ext_tensor_to_tensor_real t2)}
```
But, again, on this approach, we need to define the behavior of the add_operator when one of the inputs is a special value.

### Solution 3: Partial refinement mapping
Another approach is to define a partial refinement mapping from extended_real to Real, where we only map regular real numbers and leave special values unmapped. This means that operations resulting in special values would not have a corresponding representation in the Real domain. 