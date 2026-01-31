# Training Data Directory

Place your training data files here:

## Supported Formats:

### 1. NPZ Format (NumPy compressed)
**File:** `dataset.npz`

Contains:
- `X`: NumPy array of shape (N, 63) - N samples, each with 21 landmarks × 3 coordinates
- `y`: NumPy array of shape (N,) - Labels (e.g., 'A', 'B', 'C', ...)

### 2. JSON Format
**File:** `collected_data.json`

Array of objects with:
```json
{
  "sign_name": "A",
  "category": "alphabets",
  "landmarks": [{"x": 0.5, "y": 0.6, "z": 0.0}, ...],
  "timestamp": "2026-01-30T10:00:00"
}
```

## Your Dataset:

Place your alphabet sign language dataset here with one of the above formats.

## Training:

Once your data is here, run:
```bash
cd ../..
python ml/train_models.py
```

Or use the Windows batch script:
```bash
train_models.bat
```
