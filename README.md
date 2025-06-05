# ALINet Malaria Detection API

This project provides a FastAPI-based REST API for malaria detection using the custom ALINet deep learning model.

## Features

- Predicts malaria infection from uploaded blood smear images.
- Returns prediction, confidence, and class probabilities.
- Serves a simple HTML frontend (`index.html`).

## Requirements

- Python 3.8+
- pip

### Main Dependencies

- fastapi
- uvicorn
- torch
- torchvision
- pillow
- numpy

Install dependencies:

pip install -r requirements.txt

2. Run the API:

univocorn main:app

3. Open [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

- `GET /` — Returns the main HTML page.
- `POST /predict` — Accepts an image file and returns prediction results.

## File Structure

- `main.py` — Main API code and model definition.
- `index.html` — Frontend page.
- `alinet_best_model.pth` — Trained model weights.

## Notes

- The model expects RGB images of size 128x128.
- Supported classes: `Parasitized`, `Uninfected`.

## License

MIT
