# 📄 Documentation: `emulator/prober.py`

## 1. Overview & Architecture
`emulator/prober.py` implements the **Physical Representation Probing Suite**.

In representation probing, linear or non-linear models (probes) are trained on intermediate neural representations $h_l$ to predict ground truth physical properties:
$$h_l \longrightarrow \text{Physical Target}$$
By evaluating predictive accuracy ($R^2$, RMSE, Pearson correlation) on held-out test data, probing measures **how much physical information is linearly accessible within each layer of the model**.

---

## 2. Class: `PhysicalProbeSuite`

### Probed Physical Targets
The probe suite evaluates 7 key physical quantities:
1. `total_energy`: Total conserved energy $E = K + U$.
2. `angular_momentum_norm`: Total angular momentum magnitude $\|\vec{L}\| = \|\sum \vec{r}_i \times \vec{p}_i\|$.
3. `angular_momentum_vec`: Full 3D angular momentum vector $(L_x, L_y, L_z)$.
4. `linear_momentum_norm`: Total linear momentum magnitude $\|\vec{P}\| = \|\sum m_i \vec{v}_i\|$.
5. `linear_momentum_vec`: Full 3D linear momentum vector $(P_x, P_y, P_z)$.
6. `positions`: Raw Cartesian coordinates of all bodies $(\mathbf{p}_1, \ldots, \mathbf{p}_N)$.
7. `velocities`: Raw velocity vectors of all bodies $(\mathbf{v}_1, \ldots, \mathbf{v}_N)$.
8. `kinetic_energy`: Total kinetic energy $K = \sum \frac{1}{2} m_i v_i^2$.
9. `potential_energy`: Total potential energy $U = U_{\text{BH}} + U_{\text{self}}$.

---

## 3. Core Methods & Metrics

### Method: `extract_target_arrays(physics_list)`
Converts the list of per-step physical metadata dictionaries into batched NumPy matrices of shape $(N_{\text{samples}}, D_{\text{target}})$.

### Method: `evaluate_representation(train_h, test_h, train_targets, test_targets, probe_type)`
Fits a probe mapping normalized representations to normalized physical targets and evaluates generalization metrics on held-out test data.

#### Probing Models
- **Linear Probe (`probe_type="linear"`)**:
  Fits a cross-validated Ridge regression model (`RidgeCV(alphas=[1e-4, 1e-2, 0.1, 1.0, 10.0])`).
  Linear decodability indicates that the physical quantity is explicitly represented as a linear subspace within the hidden activation space.
- **MLP Probe (`probe_type="mlp"`)**:
  Fits a Multi-Layer Perceptron (`hidden_layer_sizes=(64,)`, ReLU activation) to detect non-linear or implicitly encoded physical relationships.

#### Evaluation Metrics
1. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   - $R^2 = 1.0$: Perfect physical decoding.
   - $R^2 = 0.0$: Performance equivalent to predicting the mean value.
   - $R^2 < 0.0$: Poor generalization / severe overfitting.
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Normalized RMSE (NRMSE)**:
   $$\text{NRMSE} = \frac{\text{RMSE}}{\sigma_y}$$
   Measures relative prediction error scaled by the variance of the true physical quantity.
4. **Pearson Correlation Coefficient ($r$)**:
   $$r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}$$
   Measures linear alignment between predicted and true physical values regardless of scale shifts.
