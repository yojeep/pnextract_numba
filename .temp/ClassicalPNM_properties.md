# Pore-Network Model Parameters: Mathematical Definitions

## Notation

| Symbol                                     | Meaning                                                                       |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| $ R_p $                                  | Radius of pore$ p $                                                         |
| $ R_t $                                  | Radius of throat$ t $                                                       |
| $ \mathbf{x}_p $                         | Center coordinate of pore$ p $ (from its medial ball)                       |
| $ \mathbf{x}_t $                         | Center coordinate of throat$ t $ (from its medial ball)                     |
| $ \mathbf{C}_t = (C_x, C_y, C_z) $       | Discrete cross-sectional vector of throat$ t $                              |
| $ A_t = \|\mathbf{C}_t\| $               | Effective cross-sectional area of throat$ t $                               |
| $ V_p $                                  | Volume of pore$ p $                                                         |
| $ S_p $                                  | Surface area of pore$ p $                                                   |
| $ l_{pt}^{(1)}, l_{pt}^{(2)} $           | Distances from throat center to centers of adjacent pores                     |
| $ L_{12} = l_{pt}^{(1)} + l_{pt}^{(2)} $ | Total center-to-center distance between two connected pores via throat$ t $ |
| $ G_t $                                  | Shape factor of throat$ t $                                                 |
| $ G_p $                                  | Effective shape factor of pore$ p $                                         |
| $ \mathcal{T}(p) $                       | Set of throats connected to pore$ p $                                       |

---

## Throat Parameters

### 1. Throat Center and Radius

The **throat center** is defined as the center of the medial ball associated with the throat:

$$
\mathbf{x}_t = \big( \texttt{tr.mb22()->fk},\ \texttt{tr.mb22()->fj},\ \texttt{tr.mb22()->fi} \big)
$$

This point lies on the medial axis and represents the geometric “skeleton” location of the narrowest passage between two pores.

The **throat radius** is bounded by the medial ball at the throat and the radii of its two adjacent pores:

$$
R_t = \min\left( \max(R_t^{\text{mb}},\, 0.5),\, R_{p_1},\, R_{p_2} \right) + \delta
$$

- $ R_t^{\text{mb}} $: radius of the maximal inscribed sphere (medial ball) at the throat
- $ \delta \sim \mathcal{U}(-0.25,\, +0.25) $: small random perturbation for numerical stability
  - Implemented as:
    $$
    \delta = 0.5 \times \left(0.5 - \frac{\texttt{rand()}}{\texttt{RAND\_MAX}}\right)
    $$

---

### 2. Cross-Sectional Vector $\mathbf{C}_t$ and Area $A_t$

During network construction (`createNewThroats`), for every pair of adjacent voxels belonging to different pores $p_1$ and $p_2$, the code detects a face interface and accumulates a signed count into the throat’s `CrosArea` vector:

- If the interface is perpendicular to the **x-axis**, then:
  $$
  C_x \leftarrow C_x + (2 \cdot \mathbb{I}[p_2 > p_1] - 1)
  $$
- Similarly for **y-axis** → $C_y$, and **z-axis** → $C_z$.

Here, $\mathbb{I}[\cdot]$ is the indicator function, and the sign ensures symmetry: swapping $p_1$ and $p_2$ flips the sign of $\mathbf{C}_t$, but $A_t = \|\mathbf{C}_t\|$ remains unchanged.

The **effective cross-sectional area** is then:

$$
A_t = \|\mathbf{C}_t\| = \sqrt{C_x^2 + C_y^2 + C_z^2}
$$

> **Note**: $A_t$ is dimensionless and proportional to the number of voxel faces connecting the two pores. It serves as a discrete proxy for hydraulic conductance area.

---

### 3. Pore  Lengths and Throat Length

Distance from throat center to each adjacent pore center:

$$
l_{pt}^{(1)} = 
\begin{cases}
x_t & \text{if } p_1 \text{ is inlet (index 0)} \\
L_x - x_t & \text{if } p_1 \text{ is outlet (index 1)} \\
\|\mathbf{x}_{p_1} - \mathbf{x}_t\| & \text{otherwise}
\end{cases}
$$

Similarly for $ l_{pt}^{(2)} $. Total inter-pore distance:

$$
L_{12} = l_{pt}^{(1)} + l_{pt}^{(2)}
$$

If $ L_{12} < 3 $, it is clamped: $ L_{12} \leftarrow 3.01 $

Each pore is assigned 67% of its half-length (except boundary pores):

$$
l_{p_1} = 
\begin{cases}
1 & \text{if } p_1 \text{ is boundary} \\
0.67 \cdot l_{pt}^{(1)} & \text{otherwise}
\end{cases}, \quad
l_{p_2} = 
\begin{cases}
1 & \text{if } p_2 \text{ is boundary} \\
0.67 \cdot l_{pt}^{(2)} & \text{otherwise}
\end{cases}
$$

Throat segment length:

$$
l_t = \max\left( L_{12} - l_{p_1} - l_{p_2},\, 10^{-7} \right)
$$

---

### 4. Throat Shape Factor

Defined as:

$$
G_t = \frac{R_t^2}{4 A_t}
$$

To ensure physical plausibility, bounds are enforced:

$$
G_t \leftarrow
\begin{cases}
\min\left(0.079,\, G_t / 2\right) & \text{if } G_t \geq 0.09 \\
\max\left(G_{\text{rand}},\, 0.01\right) & \text{if } G_t < 0.01
\end{cases}
$$

- $ G_{\text{rand}} $: small random value (e.g., drawn from uniform distribution)

---

## Pore Parameters

### 1. Pore Radius

$$
R_p = \max(R_p^{\text{mb}},\, 1.0)
$$

- $ R_p^{\text{mb}} $: radius of the medial ball associated with pore $ p $

---

### 2. Pore Shape Factor

Area-weighted average of connected throat shape factors:

$$
G_p = \frac{\sum_{t \in \mathcal{T}(p)} G_t A_t}{\sum_{t \in \mathcal{T}(p)} A_t}
$$

---

### 3. Pore Volume Adjustment

An effective cross-sectional area for the pore is estimated as:

$$
A_p = \frac{R_p^2}{4 G_p}
$$

Total weighting area:

$$
A_{\text{total}} = \sum_{t \in \mathcal{T}(p)} A_t + A_p
$$

Let $ V_p^{\text{orig}} $ be the original geometric volume of pore $ p $ (counted during voxel labeling). Then:

- Adjusted pore volume:

  $$
  V_p = V_p^{\text{orig}} \cdot \frac{A_p}{A_{\text{total}}}
  $$
- Each connected throat $ t $ receives a volume share:

  $$
  V_t \mathrel{+}= V_p^{\text{orig}} \cdot \frac{A_t}{A_{\text{total}}}
  $$

This ensures total volume conservation while redistributing pore volume based on hydraulic relevance.
