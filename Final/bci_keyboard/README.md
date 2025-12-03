# T9 Keyboard with Prediction

A web-based T9 keyboard interface with automatic scanning and word prediction.

## Project Structure

```
t9_keyboard/
│
├── app.py              # Flask server + T9 keyboard logic
│
├── static/
│   ├── style.css       # UI styling
│   └── script.js       # Frontend logic (polling & display)
│
└── templates/
    └── index.html      # HTML structure
```

## How It Works

### Backend (Python/Flask)
- Runs T9 keyboard logic with automatic scanning
- Provides API endpoints:
  - `GET /` - Serves the HTML interface
  - `GET /api/state` - Returns current keyboard state (JSON)
  - `POST /api/input/<state>` - Receives state inputs (1, 2, or 3)

### Frontend (HTML/CSS/JavaScript)
- Polls `/api/state` every 100ms for updates
- Displays:
  - Output text
  - Current word being typed
  - Word predictions
  - Current scanning position (highlighted)
- Sends state inputs via button clicks or keyboard (1, 2, 3 keys)

### State Controls

- **State 1**: Cancel/return to key scanning (useful when in prediction mode)
- **State 2**: Confirm selection (select key → letter → or prediction)
- **State 3**: Enter prediction mode (if predictions are available)

## Running the Application

1. Install Flask:
   ```bash
   pip install flask
   ```

2. Run the server:
   ```bash
   python app.py
   ```

3. Open browser to: `http://localhost:5000`

4. Use buttons or keyboard keys (1, 2, 3) to control the keyboard

## How to Use

1. The keyboard automatically scans through keys (1-9)
2. Press **State 2** when the desired key is highlighted
3. It will scan through letters on that key
4. Press **State 2** again to select the letter
5. As you type, predictions appear
6. Press **State 3** to enter prediction scanning mode
7. Press **State 2** to select a prediction
8. Press **State 1** to cancel and return to key scanning

## Future Integration

Currently uses keyboard input to simulate ML states. In the future:
- ML model will output states (1, 2, 3) based on external input
- Replace the `/api/input/<state>` endpoint calls with ML model outputs
- The rest of the system remains the same!

## Customization

- **Scan speed**: Change `scan_speed` parameter in `app.py` (default: 0.8 seconds)
- **Poll interval**: Change `POLL_INTERVAL` in `script.js` (default: 100ms)
- **Word list**: Modify `_get_hardcoded_words()` in `app.py`
- **Styling**: Edit `static/style.css`