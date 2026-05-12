# KwenaRachelapa-intern-challenge1
# Challenge 1: Fix My Broken Model

## Goal
Fix a broken telecom complaint classifier and remove data leakage issues.

## Problem
Original model showed 94% accuracy but failed in real use due to:
- data leakage from complaint_id
- TF-IDF fitted before train/test split
- incorrect evaluation setup
- wrong label mapping

## Fixes
- removed leaked features
- fixed preprocessing order
- corrected TF-IDF workflow
- improved evaluation metrics
- fixed label mapping

## Result
Model now uses real text patterns and gives reliable evaluation.
