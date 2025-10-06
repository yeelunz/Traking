# Preprocessing Modules

This directory contains pluggable preprocessing modules registered via `register_preproc`.

## Added Modules

### SRAD (`SRAD`)
Simplified Speckle Reducing Anisotropic Diffusion for speckle noise suppression while preserving edges.
Config: `iterations` (int), `lambda` (time step <=0.25), `eps`, `convert_gray`.

### Log / Dynamic Range Compression (`LOG_DR`)
Per-channel or luminance dynamic range compression by logarithmic or gamma mapping.
Config: `method` ('log'|'gamma'), `gamma`, `clip_percentile`, `per_channel`.

### Time Gain Compensation (`TGC`)
Depth (vertical) gain profile to compensate attenuation. Modes: linear / exp / custom.
Config: `mode`, `gain_start`, `gain_end`, `exp_k`, `custom_points`, `per_channel`, `clip`.

## Example Pipeline Snippet
```yaml
experiments:
  - name: exp_with_preproc
    pipeline:
      - type: preproc
        name: SRAD
        params: { iterations: 5, lambda: 0.15 }
      - type: preproc
        name: LOG_DR
        params: { method: log, clip_percentile: 99.0 }
      - type: preproc
        name: TGC
        params: { mode: linear, gain_end: 2.5 }
      - type: model
        name: YOLOv11
        params: { pretrained: true }
```
