# ASJM Channel Predictor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

A practical web tool for **process engineers and researchers** working with
abrasive slurry jet micro-machining (ASJM) of channels.
Plan and verify a channel **before you cut**:

- **Predict the channel** that a given set of process parameters will produce
  (pump pressure, stand-off distance, traverse speed, number of passes).
- **Recommend process parameters** that will produce a desired channel depth and
  width, ranked from best to worst, with an uncertainty estimate so you know
  whether the recipe is unique or whether several physically distinct recipes can
  hit the same target.

The underlying surrogate is a physics-informed artificial neural network (PI-ANN)
trained on 270 measured channel profiles. Inference takes ~100 ms per query on
an ordinary CPU, so the inverse search returns recommendations in under a second.

---

## Operating envelope

The model is **only valid inside the trained envelope**:

| Parameter                  | Range          | Units  |
| -------------------------- | -------------- | ------ |
| Pump pressure (P)          | 193 – 275      | MPa    |
| Stand-off distance (SOD)   | 1 – 5          | mm     |
| Traverse speed (V)         | 500 – 1500     | mm/min |
| Number of passes (N)       | 1 – 10         | passes |

Targets outside this envelope are clamped to the boundary; if the predicted
geometry is far from your target, your target may simply be unreachable on the
machine and abrasive used in the training set.

---

## How to use it

### Forward mode (predict a channel)
1. Open the app, choose **Forward** in the sidebar.
2. Move the four sliders to your intended process parameters.
3. The cross-section updates instantly. The KPI cards report depth (µm), width
   FWHM (µm), aspect ratio, and the AR bin (low / medium / high).

### Inverse mode (find process parameters)
1. Choose **Inverse** in the sidebar.
2. Enter the **target depth** and **target width** in micrometres.
3. Pick how many candidates *K* you want and the search budget.
4. Click **Search candidate parameters**.
5. Read off:
   - the **best candidate** P / SOD / V / N at the top,
   - the **top-K table** if you want alternatives,
   - the **spread / uncertainty** table — small std means a unique recipe,
     large std means several distinct recipes can hit the same target and you
     should choose the one that best suits your secondary objectives
     (cycle time, abrasive consumption, machine wear).

### Interpreting outputs
- **Depth** is taken from the predicted profile minimum.
- **Width** is the full width at half maximum depth (FWHM).
- The PI-ANN was trained with a monotonicity penalty so that increasing N never
  decreases the predicted depth — useful for multi-pass planning.
- The dashed orange line in inverse mode marks your target depth so you can see
  by eye how close the recommended recipes get.

---

## Recommended workflow before production

1. Use **Inverse** to get a short-list of 3–10 candidate recipes.
2. Pick the recipe that best matches your secondary constraints (cycle time,
   pressure rating, abrasive cost).
3. Use **Forward** with that recipe to inspect the predicted profile shape.
4. Cut a single pilot channel on the actual machine to confirm; tune from there.

The model is a planning aid; always validate with a pilot cut before production.

---

## Run locally

```bash
git clone https://github.com/Safaei-Fatemeh/asjm-channel-predictor.git
cd asjm-channel-predictor

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# or: source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

---

## Files

| File                                | Purpose                                                |
| ----------------------------------- | ------------------------------------------------------ |
| `app.py`                            | Streamlit web application                              |
| `src/paper_utils.py`                | Feature engineering and channel-geometry utilities     |
| `forward_pi_ann.h5`                 | Trained physics-informed ANN forward surrogate         |
| `forward_pi_scalers.npz`            | Input / output standardisation parameters              |
| `Experimental_profiles_fitted.csv`  | 270-sample experimental dataset (process params + profiles) |
| `requirements.txt`                  | Python dependencies                                    |

---

## Citation

If this tool helps your work, please cite the accompanying paper
(full reference will be added once the paper is published).

## License

The code is released for non-commercial research and educational use.
For commercial use, please contact the authors.
