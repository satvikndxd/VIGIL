# VIGIL arrhythmia research summary

## Problem
Beat-level multiclass ECG arrhythmia classification over AAMI-style N/S/V/F/Q research groups.

## Dataset
Kaggle mirror `abdallahwagih/mit-bih-arrhythmia-database` of the MIT-BIH Arrhythmia Database. Records are read as WFDB signals and expert annotations.

## Leakage control
Subject/record groups are split before beat construction and no beat from a held-out group enters training. Temporal sequences use only the current and preceding beats within each record.

## Preprocessing
Channel 0 at 360 Hz; 90 samples before and after each annotation; per-beat robust median/MAD normalization; clipping at +/-8; sequence length 8.

## Models
Logistic Regression, Random Forest, RNN, LSTM, GRU, BiLSTM, and BiLSTM with temporal attention. Class-weighted cross entropy is used for the neural models.

## Measured results
{results[['Accuracy','BalancedAccuracy','MacroF1','AUROC_OVR','AUPRC_macro','Brier_multiclass','ECE']].round(4).to_string()}

## Explainability and limitations
Attention and Integrated Gradients are descriptive analyses; neither establishes causality. Dataset shift, record-level morphology, annotation uncertainty, class imbalance, limited subject count, and the Kaggle mirror provenance limit generalization. This is not a clinical diagnostic product.

## Reproducibility and deployment
The checkpoint, label mapping, preprocessing configuration, inference bundle, predictions, plots, metrics, and training histories are exported under `artifacts_arrhythmia/`.
