# XrayTrust — Streamlit Deployment Guide

## Folder structure (must match exactly)

```
xraytrust_app/
├── app.py
├── trust_utils.py
├── requirements.txt
├── README.md
└── models/
    ├── generator_5shot.pth
    ├── generator_10shot.pth
    ├── generator_20shot.pth
    ├── generator_40shot.pth
    ├── generator_100shot.pth
    └── generator_200shot.pth
```

You only need the **generator** .pth files (not the discriminator) — the
app only does inference, not training.

## Step 1 — Before uploading anything: fix the Generator class

Open `app.py` and find the `class Generator(nn.Module):` block near the
top. **Replace it with the exact same Generator class you used in your
training notebook.** If the architecture doesn't match exactly, loading
the `.pth` weights will fail with a "size mismatch" or "missing keys"
error — this is the single most important step.

## Step 2 — Test locally first (recommended)

```bash
cd xraytrust_app
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL it prints (usually `http://localhost:8501`). Upload a
test LR image and confirm it runs without errors before deploying.

## Step 3 — Push to GitHub

```bash
cd xraytrust_app
git init
git add .
git commit -m "Initial XrayTrust Streamlit app"
git branch -M main
git remote add origin https://github.com/Imon39/xraytrust-app.git
git push -u origin main
```

(Create the empty repo first on github.com under your account, then use
its URL in the `git remote add` line above.)

### About the model file sizes

Generator `.pth` files can be large. GitHub blocks files over 100 MB by
default.

- If each `.pth` file is **under 100 MB**: a normal `git add`/`push` works fine.
- If any file is **over 100 MB**: use [Git LFS](https://git-lfs.com/):
  ```bash
  git lfs install
  git lfs track "models/*.pth"
  git add .gitattributes
  git add models/*.pth
  git commit -m "Add model weights via LFS"
  git push
  ```

## Step 4 — Deploy on Streamlit Community Cloud (free)

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repo (`Imon39/xraytrust-app`), branch `main`, and set the
   main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL like:
   `https://xraytrust-app-<random>.streamlit.app`
5. This URL is what you demo at the Innovation Fair — it works on any
   device with a browser, no local setup needed for the judges.

## Step 5 — What to say if judges ask about the "API"

This app doesn't call an external API — the model runs directly inside
the Streamlit Cloud container using the same PyTorch generator you
trained. If you're specifically asked about API-based deployment (e.g.
FastAPI + separate frontend), that's a valid next step but not required
for this prototype — the current setup already demonstrates the full
reconstruction + trust-score pipeline end-to-end.

## Common issues

| Problem | Likely cause |
|---|---|
| `RuntimeError: Error(s) in loading state_dict` | Generator class in `app.py` doesn't match your training notebook's class exactly |
| App builds but crashes on upload | Input image isn't grayscale/square — the app auto-converts, but very unusual formats (e.g. multi-page TIFF) may need extra handling |
| "Model file not found" | `.pth` files not in `models/` folder in the repo, or filename doesn't match `generator_{shot}shot.pth` |
| App works locally but not on Streamlit Cloud | Usually a `requirements.txt` version mismatch — check the "Manage app" logs on share.streamlit.io for the exact error |
