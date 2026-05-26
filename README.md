# FitAI — Privacy-Preserving Personalized Fitness Planning

A full-stack fitness planning application combining **Federated Learning**, **Personalized Federated Learning (pFL)**, and an **AI Agent** to deliver private, individualized fitness and nutrition plans — without ever exposing raw user health data.

---

## Overview

Most fitness apps suffer from two problems: they centralize sensitive health data, and they serve generic, one-size-fits-all plans. FitAI solves both.

- Raw data never leaves the client (Federated Learning via Flower/gRPC)
- Plans are personalized to each user's physiology (pFL fine-tuning)
- Every prediction is mathematically private (Differential Privacy, ε = 5.0)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, Tailwind CSS, Vanilla JS (11 pages) |
| Backend | Flask REST API (18 endpoints), JWT auth |
| FL Framework | Flower (flwr) over gRPC, 3 simulated gym clients |
| Database | SQLite with AES-256 encryption at rest |
| AI Agent | Google Gemini 2.5 Flash Lite |
| Privacy | Laplace DP at inference + Federated Learning + AES-256 |

---

## ML Models

All three models are trained federally across distributed clients:

| Model | Type | Performance |
|---|---|---|
| Weight Prediction | GradientBoosting + pFL fine-tuning | MAE = 0.339 kg, R² = 0.9986 |
| Adherence Classifier | Random Forest (binary) | 86.7% accuracy |
| Macro Recommendation | GradientBoosting | R² = 0.871 |

Differential Privacy adds an average of only **0.04 kg** noise per prediction — negligible accuracy cost.

---

## AI Agent Architecture

A 5-phase pipeline: **Perceive → Reason → Plan → Act → Adapt**

FL model outputs (weight prediction, adherence score, macro targets) are passed as structured inputs to Gemini — not raw user data — to generate weekly personalized fitness and nutrition plans.

---

## Privacy Stack (3 layers)

1. **Federated Learning** — raw data never leaves the client
2. **AES-256 encryption at rest** — all sensitive DB fields encrypted
3. **Differential Privacy (Laplace, ε = 5.0)** — applied live at every inference

Satisfies **GDPR Article 25 (Privacy by Design)** requirements.

---

## Food Database

~1,800 items sourced from **USDA FoodData Central** and **IFCT 2017**, standardized at 100g per entry, covering both international and Indian cuisine with real macro values.

---

## Key Results

- pFL model achieves **59.2% lower MAE** than basic global FL
- Tested with real users across a complete, deployed application
- Weekly HTML email summaries delivered post-plan generation

---

## Project Info

**Degree:** B.Tech in Data Science and Engineering  
**Institution:** Manipal University Jaipur  
**Author:** Swapnil Mogal (229309019)  
**Guide:** Dr. Meenakshi Gaur  
**Year:** 2025–26
