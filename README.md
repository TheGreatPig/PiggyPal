# PiggyPal
Simple web-based budget app
Import Spending Information (csv file) and see what you're spending your money on with insightful statistics.
Categorization of transactions using tiny llm

---

## Project Plan

### Tech Stack
- **Frontend:** SvelteKit
  - CSV upload UI
  - Statistics display using Chart.js
- **Backend:** FastAPI
  - Receives CSV uploads
  - Processes and parses CSV data
  - Integrates with ONNX runtime for LLM-based transaction categorization
  - Returns categorized data and statistics to frontend
- **LLM:** Tiny model running via ONNX for efficient, server-side inference

### Workflow
1. **User uploads a CSV file** via the SvelteKit frontend.
2. **Frontend sends the file** to the FastAPI backend.
3. **Backend parses the CSV** and uses the ONNX LLM to categorize each transaction.
4. **Backend computes statistics** (totals, categories, trends, etc.).
5. **Backend returns results** to the frontend.
6. **Frontend displays statistics** and charts using Chart.js.

### Security & Privacy
- Uploaded files are processed in-memory and deleted after analysis.
- No persistent storage of user data in MVP.
- Rate limiting and validation to prevent abuse.

### Deployment
- SvelteKit frontend and FastAPI backend can be deployed together or separately (e.g., Vercel/Netlify for frontend, cloud/VPS for backend).
- ONNX model hosted alongside backend for fast inference.

---

This plan covers the MVP. Future enhancements may include user accounts, persistent storage, and more advanced analytics.


# Backend

This is the FastAPI backend for PiggyPal.

## Run the server:
   ```bash
   uvicorn main:app --reload
   ```
