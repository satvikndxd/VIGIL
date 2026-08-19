Workbench UI validation:

The redesigned dashboard at http://127.0.0.1:5173 now renders a SQL-workbench composition: dark SQL/analysis editor, light query-result table, connections tree, worksheet output, ECG waveform, probability bars, temporal attention, and a result panel. Live values include the RNN prediction V at 74.3% confidence, measured benchmark values, and actual held-out rows.

The Experiments view renders the existing model benchmark as a query-result table plus compact Macro-F1 result cards for Logistic Regression, Random Forest, RNN, LSTM, GRU, BiLSTM, and BiLSTM_Attention. It keeps measured results visible rather than forcing the attention model to appear best.
