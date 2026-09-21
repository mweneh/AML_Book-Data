# Lab 08 — Recurrent Neural Networks: Forecasting COVID-19 Cases

Lab 08 forecasts daily new COVID-19 cases in the US from `Data/covid_19_data.csv`.

## Preparing the series

The raw file has one row per country/region per day with a **cumulative** confirmed-case
count. The preparation is: keep US rows and sum them (some days have several state rows),
take the day-to-day difference to convert cumulative counts into new cases per day, clip
negative values to zero (counts are sometimes corrected downward), and smooth with a 7-day
rolling average because reporting is uneven across the week — fewer cases are reported at
weekends.

## Windowing

A time series is turned into supervised `(X, y)` pairs by sliding a window: the previous
`LOOKBACK = 30` days are the input and the next day is the target. This is the step that
makes forecasting a supervised learning problem.

## Splitting

The last 60 days are held out as the test set. The split is **by date, never randomly** — a
random split would let the model train on future values and predict the past, which is
leakage. Scaling uses `MinMaxScaler` fitted on the training portion only.

## SimpleRNN versus LSTM

A SimpleRNN reads the sequence one step at a time, carrying a hidden state forward. Its
weakness is the vanishing gradient: over a long sequence the gradient signal shrinks
multiplicatively as it is backpropagated through time, so the network cannot learn
dependencies more than a few steps back.

An LSTM adds a **cell state** plus three gates — forget, input and output — that control
what is discarded, what is written and what is read out. The cell state gives the gradient
an additive path through time instead of a multiplicative one, which is why the LSTM learns
long-range dependencies that a SimpleRNN cannot. A GRU is a lighter variant with two gates
and no separate cell state.

## Baseline

Both models are compared against a naive baseline that predicts "tomorrow equals today".
This baseline is surprisingly hard to beat on a smoothed series, and any forecasting model
that does not beat it has demonstrated nothing. The reported metric is MAE in cases per day.
