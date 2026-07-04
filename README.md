# Post-Judgment Compensation Ratio Reference Estimator

This Streamlit app uses a logistic regression model to provide a non-binding, model-based reference estimate for post-judgment compensation ratio assessment in de-identified medicolegal cases.

## Files

- `app.py`: Streamlit app
- `model.pkl`: trained logistic regression model bundle
- `requirements.txt`: Python dependencies

## Local run

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

## Disability severity encoding

Users enter the actual disability severity grade from 1 to 10.

Backend encoding rule:

- Grades 1–4 are encoded as 1
- Grades 5–10 are encoded as 0

## Output

The app shows:

- predicted high-risk probability
- predicted risk level based on threshold 0.5

## Privacy and use notice

When users enter the interface, an Important Notice dialog is displayed before any case-level information can be entered. Users must confirm that they understand the privacy and use limitations before continuing.

Users should not enter real names, case numbers, medical record numbers, ID numbers, addresses, contact information, or any identifiable personal information.

The authors do not store user inputs or use them for model retraining. However, this application is hosted on a third-party cloud platform, which may process standard technical logs or metadata according to its own policies.

This tool provides only a non-binding, model-based reference estimate for research and auxiliary use. It is not legal advice, medical advice, judicial appraisal, liability determination, or a basis for court decisions.
