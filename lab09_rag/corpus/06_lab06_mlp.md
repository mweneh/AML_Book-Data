# Lab 06 — Neural Networks: the Multi-Layer Perceptron

Lab 06 is the first deep learning lab. It builds an MLP in Keras on the tabular loan data
and compares it against the gradient-boosted baseline from Lab 04.

## Architecture

The network is `Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu) -> Dropout(0.3) ->
Dense(1, sigmoid)`. Inputs are standardised with `StandardScaler` fitted on the training
split only.

## Why ReLU

ReLU is `max(0, x)`. It is preferred over sigmoid and tanh in hidden layers because its
gradient is exactly 1 for positive inputs, so it does not saturate and does not squash the
gradient signal as it flows backward through many layers. Sigmoid saturates at both ends,
where its gradient approaches zero — this is the **vanishing gradient problem**.

The known failure mode of ReLU is the "dying ReLU": a unit whose input is always negative
outputs zero forever and its weights stop updating. Leaky ReLU is the usual remedy.

## Loss and optimiser

Binary classification uses `binary_crossentropy`. The optimiser is Adam with a learning
rate of 0.001. Adam adapts a per-parameter learning rate using running estimates of the
first and second moments of the gradient, which makes it much less fussy to tune than plain
SGD.

## Regularisation

Three mechanisms are used together: dropout at 0.3, L2 weight decay, and early stopping
with `patience=10` and `restore_best_weights=True`. Early stopping monitors validation loss
and is the cheapest form of regularisation available — it costs nothing and requires no
tuning beyond the patience value.

## Class imbalance

`class_weight` is passed to `model.fit()` to up-weight the minority class, which is the
neural-network equivalent of `scale_pos_weight` in XGBoost.

## The honest conclusion

The MLP reaches PR-AUC roughly comparable to the tuned gradient-boosted model but takes far
longer to train and is harder to interpret. **On tabular data, gradient boosting is usually
still the right default.** Neural networks earn their keep on unstructured data — images,
audio, text and sequences — which is what Labs 07, 08 and 09 cover.
