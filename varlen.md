$$
\begin{align}
    out &= C\hat{H}I \\
    l &= 0,\, 1, \, ... \, maxDelays =: D \\
    C &= \begin{pmatrix}
            1_0 & 0_1 & \cdots & 1_{2D-1} & 0_{2D} \\
            0_0 & 1_1 & \cdots & 0_{2D-1} & 1_{2D} \\
        \end{pmatrix} \\
    \hat{H} &= diag_2(H_l) \\
            &=
        \begin{pmatrix}
            H_0     & O_2   &           & \cdots    &       &  O_2      \\
            O_2     & H_1   &           & \cdots    &       &  O_2      \\
            \vdots  &       &  \ddots   &           &       &  \vdots   \\
                    &       &           & H_l       &       &           \\
            O_2     &       &           & \cdots    &       &  H_{Delays}    
        \end{pmatrix}\\
    H_l &= \begin{pmatrix}
            a_l & b_l \\
            c_l & d_l \\
            \end{pmatrix} \\
    H &= \begin{pmatrix}
            a_0 & b_0 & 0   & 0   & \cdots & 0 \\
            c_0 & d_0 & 0   & 0   & \cdots & 0 \\
            0   & 0   & a_1 & c_1 & \cdots & 0 \\
            0   & 0   & b_1 & d_1 & \cdots & 0 \\
            \vdots & \ddots & & & & \vdots     \\
            0   & 0   & 0   & 0   & a_D & b_D  \\
            0   & 0   & 0   & 0   & c_D & d_D  \\
        \end{pmatrix} \\
    I &= \begin{pmatrix}
            x_{0,0,0} & x_{0,0,1} & \cdots & x_{0,0,samples}  \\
            x_{0,1,0} & x_{0,1,1} & \cdots & x_{0,1,samples}  \\
            \\
            x_{1,0,0} & x_{1,0,1} & \cdots & x_{1,0,samples}  \\
            x_{1,1,0} & x_{1,1,1} & \cdots & x_{1,1,samples}  \\
            \\
            x_{2,0,0} & x_{2,0,1} & \cdots & x_{2,0,samples}  \\
            \vdots    & \vdots    &        & \vdots           \\
            x_{l,0,0} & x_{l,0,1} & \cdots & x_{l,0,samples}  \\
            x_{l,1,0} & x_{l,1,1} & \cdots & x_{l,1,samples}  \\
            \vdots    & \vdots    &        & \vdots           \\
            x_{D,0,0} & x_{D,0,1} & \cdots & x_{D,0,samples}  \\
            x_{D,1,0} & x_{D,1,1} & \cdots & x_{D,1,samples}  \\
        \end{pmatrix} \\
    C\hat{H} &= \begin{pmatrix}
        a_0 & c_0 & a_1 & \cdots & a_D & c_D \\
        b_0 & d_0 & b_1 & \cdots & a_D & d_D \\
    \end{pmatrix}
\end{align}
$$

$$
\begin{align}
    C\hat{H}I &= \begin{pmatrix}
            a_0 x_{0,0,0} + c_0 x_{0,1,0} + a_1 x_{1,0,0} + \cdots + a_D x_{D,0,0} + c_D x_{D,1,0} & \cdots & a_0 x_{0,0,samples} + c_0 x_{0,1,samples} + a_1 x_{1,0,samples} + \cdots + a_D x_{D,0,samples} + c_D x_{D,1,samples} \\
            b_0 x_{0,0,0} + d_0 x_{0,1,0} + b_1 x_{1,0,0} + \cdots + b_D x_{D,0,0} + d_D x_{D,1,0} & \cdots & b_0 x_{0,0,sbmples} + d_0 x_{0,1,sbmples} + b_1 x_{1,0,sbmples} + \cdots + b_D x_{D,0,sbmples} + d_D x_{D,1,sbmples}
        \end{pmatrix} \\
\end{align}
$$
