# Optimal Execution with Order Flow Imbalance Signals

An optimal execution framework that uses Order Flow Imbalance (OFI) signals derived from Level 2 order book data to minimize market impact when executing large orders.

## Key Results

| Strategy | Avg IS vs VWAP | Total Savings (4 test days) |
|----------|----------------|----------------------------|
| TWAP (baseline) | +2.40 bps | -$65,865 |
| VWAP (baseline) | +0.10 bps | -$2,619 |
| **OFI-VWAP Multiplicative** | **-0.73 bps** | **+$20,092** |
| **OFI Opportunistic** | **-2.26 bps** | **+$61,817** |

*Negative Implementation Shortfall = outperforming benchmark*

**Bottom line:** The OFI-enhanced strategies save ~$15-60k per 100k shares executed vs naive approaches.

---

## Problem Statement

**Goal:** Buy 100,000 SPY shares over a trading day while minimizing total market impact.

**Challenge:** Large orders move markets. Trading too fast increases temporary impact; trading too slow exposes you to price drift risk.

**Solution:** Use Order Flow Imbalance to identify favorable execution windows—periods where selling pressure creates better entry points for buyers.

---

## Methodology

### 1. Order Flow Imbalance (OFI) Signal

OFI measures the net order flow pressure at each price level:

```
OFI_t = Σ (ΔBid_size × 1{bid_price unchanged}) - Σ (ΔAsk_size × 1{ask_price unchanged})
```

- **Positive OFI:** Buying pressure (unfavorable for buyers)
- **Negative OFI:** Selling pressure (favorable for buyers)

We compute OFI across 10 levels of the order book and use PCA to create an integrated signal:

```
OFI_integrated = Σ w_i × OFI_i / Q_t
```

Where `w_i` are PCA weights and `Q_t` is average depth (normalization).

### 2. PCA Weights (Trained on 18 days)

| Level | Weight | Interpretation |
|-------|--------|----------------|
| 0 (best bid/ask) | +0.583 | Dominates signal |
| 1 | -0.077 | |
| 2 | -0.088 | |
| 3-9 | -0.04 to +0.00 | Diminishing importance |

**Variance explained:** 60%

### 3. Execution Strategies

#### Baselines
- **TWAP:** Equal allocation across all periods
- **VWAP:** Allocate proportional to volume

#### OFI-Enhanced (Initial Approach - Flawed)
```python
weight = (1 - α) × VWAP_weight + α × OFI_weight  # Additive
```
**Problem:** This overweighted low-volume periods (where OFI tends to be negative), increasing market impact.

#### OFI-Enhanced (Fixed Approaches)
```python
# Multiplicative: OFI tilts WITHIN volume, not INSTEAD of it
weight = VWAP_weight × (1 + α × OFI_adjustment)

# Volume Floor: Only apply OFI when volume > median
# Volume Scaled: OFI influence proportional to volume
```

### 4. Impact Model

Temporary impact follows a square-root model:

```
Impact = η × σ × (x/V)^0.5 × φ(OFI)
```

Where:
- `η` = impact coefficient (estimated from spread)
- `σ` = local volatility
- `x` = order size
- `V` = available volume
- `φ(OFI)` = OFI adjustment factor

---

## Key Finding: The Volume-OFI Trap

Initial OFI-tilted strategies **underperformed** VWAP. Diagnostic analysis revealed:

| Periods | Avg OFI | Avg Volume |
|---------|---------|------------|
| OFI-VWAP overweights | -0.68 | **14.5M** |
| OFI-VWAP underweights | +0.59 | **46.9M** |

**The signal was pushing trades into illiquid periods**, where market impact is 3x higher.

**Fix:** Multiplicative weighting preserves volume profile while tilting within it:

| Strategy | Overweight Volume | Underweight Volume |
|----------|-------------------|-------------------|
| Original (broken) | 11.7M | 34.6M |
| **Multiplicative (fixed)** | **27.2M** | 20.1M |

---

## Project Structure

```
├── prepare_execution_data.py    # Process LOB data, compute OFI, add price/volume
├── run_backtest_fast.py         # Initial backtest (identified the problem)
├── run_backtest_volume_aware.py # Fixed strategies backtest
├── ofi_diagnostic.py            # Diagnostic analysis
├── OPTIMAL_EXECUTION_MATH.md    # Mathematical framework
└── README.md
```

---

## Data

- **Source:** SPY Level 2 order book data (10 levels bid/ask)
- **Period:** October 2025 (22 trading days)
- **Frequency:** Event-level, binned to 10 seconds
- **Split:** 18 days train / 4 days test

### Features per 10-second bin:
- `mid_price`, `spread`, `microprice`
- `volume` (total depth)
- `book_imbalance`
- `OFI_0` through `OFI_9` (per-level order flow)
- `ofi_integrated` (PCA-weighted aggregate)

---

## Results by Day

| Date | VWAP | OFI-VWAP Mult | OFI Opportunistic |
|------|------|---------------|-------------------|
| 2025-10-27 | +0.09 bps | **-0.51 bps** | **-6.69 bps** |
| 2025-10-28 | +0.09 bps | **-0.15 bps** | **-3.01 bps** |
| 2025-10-29 | +0.11 bps | **-3.02 bps** | +0.35 bps |
| 2025-10-30 | +0.10 bps | +0.77 bps | +0.32 bps |
| **Average** | +0.10 bps | **-0.73 bps** | **-2.26 bps** |

---

## Usage

### 1. Prepare Data
```python
python prepare_execution_data.py
# Outputs: SPY_execution_data_full.parquet
```

### 2. Run Backtest
```python
python run_backtest_volume_aware.py
# Outputs: execution_results_volume_aware.csv
```

### 3. Run Diagnostics
```python
python ofi_diagnostic.py
# Outputs: ofi_diagnostic_data.parquet
```

---

## Requirements

```
pandas>=1.5
numpy>=1.21
scikit-learn>=1.0
scipy>=1.9
lightgbm>=3.3  # Optional, for OFI prediction model
```

---

## Mathematical Framework

### Optimization Problem

```
min   Σ_t g_t(x_t) + λ × σ² × Σ_t I_t²
s.t.  Σ_t x_t = S       (must execute S shares)
      x_t ≥ 0           (no short selling)
      x_t ≤ 0.1 × V_t   (participation limit)
```

Where:
- `g_t(x_t)` = temporary impact cost
- `I_t` = remaining inventory
- `λ` = risk aversion parameter

### OFI Adjustment

```
φ(OFI) = exp(λ_ofi × (OFI - μ) / σ)
```
- `OFI < 0` → `φ < 1` → lower cost (favorable)
- `OFI > 0` → `φ > 1` → higher cost (unfavorable)

---

## Lessons Learned

1. **Signal ≠ Strategy:** A predictive signal can lose money if implementation ignores constraints (volume, capacity)

2. **Diagnose failures:** The original OFI strategy had negative alpha. Rather than discard it, diagnosis revealed a fixable flaw.

3. **Multiplicative > Additive:** When combining signals with constraints, multiplicative blending often preserves important structure.

4. **Volume is king:** In execution, trading in liquid periods dominates signal quality.

---

## Future Improvements

- [ ] Train model to predict execution cost directly (not OFI)
- [ ] Test longer horizons (1min, 5min bins)
- [ ] Add spread prediction as secondary signal
- [ ] Implement adaptive `α` based on signal confidence
- [ ] Add permanent impact modeling
- [ ] Test on other liquid ETFs (QQQ, IWM)

---

## References

1. Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions.* Journal of Risk.
2. Cont, R., Kukanov, A., & Stoikov, S. (2014). *The price impact of order book events.* Journal of Financial Econometrics.
3. Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.

---

## License

MIT

---

## Author

[Your Name]

*Built as a quantitative research project exploring market microstructure and optimal execution.*
