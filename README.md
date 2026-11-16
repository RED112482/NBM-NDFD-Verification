# NBM / NDFD Temperature Guidance Viewer

A Mobile/Pensacola (WFO MOB) focused viewer for official NDFD and National Blend of Models temperature guidance.

## Phase 1

- NDFD minimum and maximum temperature grids
- NBM deterministic temperature extrema
- NBM percentile guidance (P10/P25/P50/P75/P90 when available)
- NBM 01Z / 07Z / 13Z / 19Z cycle comparison
- WFO MOB land, coastal marine, or combined sampling
- Interactive map and non-verification spatial statistics
- Archive-aware JSON schema designed for later lead-time and verification graphics

## Data sources

- NBM: NOAA Open Data, `noaa-nbm-grib2-pds`
- NDFD: NWS operational NDFD GRIB2
- Boundaries: NWS Reference Map services

The GitHub Actions workflow refreshes the compact viewer dataset and publishes `site/` to GitHub Pages.

## Local build

```bash
pip install -r requirements.txt
python scripts/build_guidance.py
python -m http.server -d site 8000
```

Then open http://localhost:8000.

## Design note

The generated dataset is keyed by product, valid time, NBM cycle, and guidance member. This is intentional: future verification and 6–12 hour forecast-evolution charts can use the same data without changing the frontend data model.
