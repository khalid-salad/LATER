# Cholesky Decomposition

Given a real symmetric positive definite matrix $A_{n\times n}$, write $A = LL^T$, where $L$ is lower triangular. 

## Algorithm

If $n \leq \texttt{blocksize}$, perform the decomposition by hand. Otherwise, write

$$A = \left[\begin{array}{ c | c }
    A_{11} & A_{12} \\
    \hline
    A_{21} & A_{22}
  \end{array}\right]$$

and

1) recursively perform Cholesky decomposition on $A_{11}$
2) solve $XA_{21} = A_{11}$ for matrix $X$ and set $A_{21} \gets X$
3) set $A_{22} \gets -A_{21}A_{21}^T + A_{22}$
4) recursively perform Cholesky decomposition on $A_{22}$