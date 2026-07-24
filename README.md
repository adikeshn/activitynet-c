# Predictive Maintenance NN

A machine-failure prediction system built around a **neural network written from scratch in C** (no ML frameworks — manual forward pass, backprop, and gradient descent), served through a **FastAPI** backend and visualized with a **live React dashboard** that simulates a running machine and streams predictions in real time.

Trained on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

---

## How it works

The project is three layers stacked on top of one shared C neural-network engine:

1. **`models/`** — the neural network itself, implemented in raw C (structs, matrix math, activations, backprop, weight serialization) and compiled two ways:
   - as a standalone binary (`machine_failure_net`) for training/testing from the command line
   - as a shared library (`machine_failure.so`) that Python loads via `ctypes`
2. **`api/`** — a FastAPI service that loads the compiled `.so` and exposes a `/predict` endpoint
3. **`site/`** — a React + Vite dashboard that simulates sensor readings on a virtual machine, streams them to the API, and plots live failure-risk predictions

```
Sensor readings (simulated in browser)
        │
        ▼
   FastAPI  /predict  ──►  ctypes  ──►  machine_failure.so (compiled C)
                                              │
                                    forward_pass() through
                                    two trained networks:
                                    1) failure_percent.bin  (binary risk score)
                                    2) category_model.bin   (failure type, if risky)
```

## The neural network (`models/src`)

Everything is hand-rolled — matrices, layers, neurons, activations, losses, and gradients are all plain C structs and loops (see `structs.h`, `util.c`, `evaluate.c`, `train.c`).

- **Architecture**: fully-connected feedforward net, configurable layer sizes, per-layer activation functions
- **Activations**: ReLU (hidden layers), Sigmoid, Softmax (output) — implemented with derivatives for backprop (`activations.c`)
- **Weight init**: He initialization for ReLU layers, Xavier initialization for Sigmoid/Softmax layers (`init.c`)
- **Losses**: binary cross-entropy (with class-weighting) and categorical cross-entropy (`cost.c`)
- **Training**: manual backpropagation with mini-batch gradient descent, Fisher–Yates shuffling per epoch, and a train/test split for held-out loss tracking (`train.c`)
- **Persistence**: networks are serialized to flat `.bin` files (`init.c`'s `read_net`/`write_net`) — no external format, just raw struct dumps

### Two models, two jobs

The system actually trains and serves **two separate networks** against the same five input features:

| Model | File | Output | Purpose |
|---|---|---|---|
| Failure risk | `models/failure_percent.bin` | 1 sigmoid output | Probability the machine fails |
| Failure category | `models/category_model.bin` | 5 softmax outputs | *If* risky, which failure type — Heat Dissipation, Power, Overstrain, Tool Wear, or Random Failure |

Input features (from the AI4I 2020 dataset): **air temperature, process temperature, rotational speed, torque, tool wear**. Feature means/stds are saved to `normalize.bin` at training time and reapplied to any new input before inference (`normalize`/`normalize_input` in `util.c`).

### Handling class imbalance

Machine failures are rare in the dataset, so the binary model uses a **weighted binary cross-entropy loss** (`tr_weight = 14.7`, `fl_weight = 0.52` in `python_facing.c`) to penalize missed failures far more heavily than false alarms. The categorical model is trained only on the subset of entries where a failure actually occurred, so it only ever has to discriminate between failure *types*.

### Training config (as set in `main.c`)

- Hidden layers: `32 → 16 → 5`, activations `ReLU → ReLU → Softmax`
- 150 epochs, batch size 16, learning rate 0.008
- 80/20 train/test split
- Seeded RNG (`srandom(2)`) for reproducibility

## API (`api/main.py`)

FastAPI service that wraps the compiled C library:

- `GET /` — health check
- `POST /predict` — takes sensor readings + an optional `threshold`, returns:
  - `failure_percent` — risk score from the binary model
  - `failure_class` — index into the failure-type categories (`-1` if below threshold, otherwise 0–4 from the categorical model)

The C code is compiled at container build time (see `Dockerfile`) into `api/machine_failure.so` and loaded via `ctypes`, with `MODEL_DIR` pointing at the `models/` directory so the compiled library knows where to find the `.bin` weight files.

CORS is configured for local dev (`localhost:5173`) and a deployed frontend on Vercel.

## Frontend (`site/`)

A React (Vite) dashboard (`site/src/App.jsx`) that:

- Simulates a virtual machine's sensor readings (air temp, process temp, rotation speed, torque, tool wear) drifting toward one of several **modes** — Normal, High Heat, High Load, Worn Tool, Unstable — each with its own target values and noise level (`data/simulationConfig.js`)
- Lets the user manually trigger **actions** (Boost Temp, Increase Load, Boost Speed, Wear Tool, Cool Machine, Stabilize) that perturb the simulated machine
- Polls the FastAPI `/predict` endpoint on an interval, plotting live failure-risk and sensor history with `recharts`
- Displays the predicted failure category (via `FAILURE_LABELS`) once risk crosses the adjustable threshold

## Project structure

```
.
├── api/
│   └── main.py              # FastAPI app, ctypes bridge to the .so
├── models/
│   ├── ai4i2020.csv          # AI4I 2020 training dataset
│   ├── failure_percent.bin   # trained binary risk model (weights)
│   ├── category_model.bin    # trained failure-category model (weights)
│   ├── normalize.bin         # saved feature means/stds
│   ├── makefile               # builds the standalone `machine_failure_net` binary
│   └── src/
│       ├── main.c            # train_models() / test_models() entry points
│       ├── init.c            # net construction, weight init, serialization, data loading
│       ├── train.c           # backprop training loop, batching, shuffling
│       ├── evaluate.c        # forward pass, backprop math, prediction, eval loss
│       ├── cost.c            # binary/categorical cross-entropy + derivatives
│       ├── activations.c     # ReLU, Sigmoid, Softmax + derivatives
│       ├── util.c            # matrix ops, normalization, layer<->matrix conversion
│       ├── free_struct.c     # manual memory cleanup for net/layer structs
│       ├── python_facing.c   # predict_percent()/predict_category() exposed to ctypes
│       └── headers/          # corresponding .h files
├── site/                     # React + Vite dashboard
│   └── src/
│       ├── App.jsx
│       ├── components/       # ControlPanel, LiveLineChart, MetricCard, PredictionPanel
│       ├── data/simulationConfig.js   # modes, actions, metric ranges, API URL
│       └── utils/simulation.js        # machine-state simulation logic
├── Dockerfile                # compiles the C lib + runs the FastAPI service
└── requirements.txt
```

## Running it

### Backend (Docker)

```bash
docker build -t predictive-maintenance-nn .
docker run -p 8080:8080 predictive-maintenance-nn
```

This compiles `models/src/*.c` into `machine_failure.so` inside the image and starts the API on `:8080`.

### Backend (local, without Docker)

```bash
pip install -r requirements.txt
gcc -shared -fPIC -o api/machine_failure.so \
    models/src/activations.c models/src/cost.c models/src/evaluate.c \
    models/src/free_struct.c models/src/init.c models/src/train.c \
    models/src/util.c models/src/python_facing.c \
    -Imodels/src/headers -lm
MODEL_DIR=$(pwd)/models uvicorn api.main:api --reload
```

### Retraining the C models directly

```bash
cd models
make
./machine_failure_net    # runs test_models() by default; edit main() to call train_models()
```

### Frontend

```bash
cd site
npm install
npm run dev
```

Set `VITE_API_URL` if the API isn't running at the default `http://127.0.0.1:8000/predict`.

## Tech stack

- **C** — neural network engine (matrices, layers, activations, backprop, serialization)
- **FastAPI + ctypes** — Python API layer wrapping the compiled C library
- **React + Vite + recharts** — live simulation dashboard
- **Docker** — builds the C shared library and serves the API in one image
