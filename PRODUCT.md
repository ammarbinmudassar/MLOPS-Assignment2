# VisionGuard

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing Flask, Jinja templates, HTML/CSS and TensorFlow/Keras. The user confirmed
direct HTML/CSS implementation for this simple coursework interface.

## Users

The assignment author and evaluator upload an image, inspect an AI/Real estimate
and confidence, and review saved prediction history.

## Product Purpose

Demonstrate local model inference, validated uploads and traceable results for
MLOps Assignment 2. Confidence is a model score, not proof of image origin.

## Capabilities and Constraints

PNG and JPEG uploads, configurable size limits, in-memory image processing,
local champion model loading once at startup, SQLite result metadata, a health
endpoint and debug-disabled Flask/Gunicorn startup. Preserve matching training
preprocessing. Do not retain uploaded image bytes or call DagsHub per request.

## Evidence on Hand

Real trained models and measured results are under outputs/task3. Task 4.1
includes an honest misclassification. Task 4.2 browser screenshots and HTTP
checks are saved under outputs/task4/4_2. Do not invent accuracy claims.

## Product Principles

- Keep the coursework scope and explanations simple.
- Make uncertainty visible beside the result.
- Save model/version provenance with each decision.
- Keep uploaded images out of persistent storage.
