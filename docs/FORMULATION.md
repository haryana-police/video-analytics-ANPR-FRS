# Formulation Document (living spec)

**Canonical link (keep this — the doc is actively updated):**

> https://docs.google.com/document/d/1k2KNIAj4Gpe_0jikH3_xZLYFpH1xCj1dCFLdg3WTXlA/edit?usp=sharing

## How to refresh the local snapshot

The current contents are mirrored to `docs/formulation_snapshot.txt`. To pull
the latest version of the doc at any time:

```powershell
$url = "https://docs.google.com/document/d/1k2KNIAj4Gpe_0jikH3_xZLYFpH1xCj1dCFLdg3WTXlA/export?format=txt"
Invoke-WebRequest -Uri $url -OutFile docs\formulation_snapshot.txt -UseBasicParsing
```

(No date is embedded in the snapshot on purpose — re-fetch whenever starting
work; the canonical link above always wins over the snapshot.)

## Summary (as of 2026-02-14 snapshot)

- **Goal**: video analytics system for police.
- **Two subsystems**: ANPR (vehicle number plate recognition) + FRS (facial
  recognition system).
- **Two modes**: Live (lightweight, non-LLM) and Offline (heavy LLM-based).
- **Phase 1 = non-LLM ANPR**, stack already in this repo:
  1. Plate Detection — YOLO11 plate
  2. OCR — Awiros ANPR OCR
  3. Object Tracking — ByteTrack
- **Target features**: object identification (vehicle, person, …), object
  tracking, vehicle number plate recognition + OCR.

See `docs/formulation_snapshot.txt` for the verbatim text.
