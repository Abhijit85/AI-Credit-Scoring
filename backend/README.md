
# AI Credit Scoring App

This is a full-stack AI-powered credit scoring system built with:
- **FastAPI** (backend)
- **AWS Bedrock** for LLM risk summarization
- **AWS Fraud Detector or Amazon SageMaker** for anomaly detection
- **React** (frontend with TailwindCSS and lucide-react)
- **MongoDB** (recommended for storage and optional vector indexing)

## Features

- 🧠 AI-generated credit score and breakdown
- 📊 Visual sliders and tabbed UI
- 💬 LLM-generated risk summary and suggestions
- 🚨 Anomaly detection service for suspicious applications
- 📁 Modular FastAPI backend

## Architecture

```
User Input → React UI → FastAPI (/score) → Rule-based scoring + Bedrock LLM + Anomaly detection → JSON → UI rendering
```

## Environment Variables

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

# Region hosting the text inference profile
BEDROCK_TEXT_REGION=us-west-2

# Provide one of the following for the profile
BEDROCK_TEXT_INFERENCE_PROFILE_ARN=arn:aws:bedrock:us-west-2:ACCOUNT_ID:inference-profile/my-text-profile
BEDROCK_TEXT_INFERENCE_PROFILE_ID=ip-1234567890abcdef

# Optional embedding profile
BEDROCK_EMBED_REGION=us-west-2
BEDROCK_EMBED_INFERENCE_PROFILE_ARN=arn:aws:bedrock:us-west-2:ACCOUNT_ID:inference-profile/my-embed-profile
FRAUD_DETECTOR_MODEL_ARN=arn:aws:frauddetector:us-east-1:123456789012:detector/my-detector   # if using Fraud Detector
SAGEMAKER_ENDPOINT_NAME=my-anomaly-endpoint                                                   # if using SageMaker
MONGODB_URI=mongodb://localhost:27017
```

The backend uses cross-region inference through the configured inference profile and omits any `modelId` from requests.

Ensure the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) is configured or the above variables are exported.

## Running Locally

### Backend
```bash
cd backend
# create .env and add the variables above
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # start development server
npm run build # create production build
npm start     # preview production build
```

Visit: http://localhost:5173/

