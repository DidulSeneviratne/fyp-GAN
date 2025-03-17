pip install -r requirements.txt
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
cd frontend
python3 -m http.server 8003