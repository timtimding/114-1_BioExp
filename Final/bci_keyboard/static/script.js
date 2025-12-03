// Poll interval in milliseconds
const POLL_INTERVAL = 100;

// Current state
let currentState = null;

/**
 * Fetch the current keyboard state from the server
 */
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        updateUI(state);
        currentState = state;
    } catch (error) {
        console.error('Error fetching state:', error);
    }
}

/**
 * Send a state input to the server
 */
async function sendState(stateNum) {
    try {
        const response = await fetch(`/api/input/${stateNum}`, {
            method: 'POST'
        });
        const result = await response.json();
        
        // Immediately fetch new state after input
        await fetchState();
    } catch (error) {
        console.error('Error sending state:', error);
    }
}

/**
 * Update the UI based on current state
 */
function updateUI(state) {
    // Update output text
    const outputText = document.getElementById('output-text');
    outputText.textContent = state.output_text || '(empty)';
    
    // Update current word
    const currentWord = document.getElementById('current-word');
    currentWord.textContent = state.current_word || '';
    
    // Update predictions
    const predictionsContainer = document.getElementById('predictions');
    const predictionModeLabel = document.getElementById('prediction-mode');
    
    // Update mode label (only if element exists)
    if (predictionModeLabel) {
        if (state.prediction_mode === 'completion') {
            predictionModeLabel.textContent = '(Word Completion)';
            predictionModeLabel.style.color = '#3498db';
        } else if (state.prediction_mode === 'next_word') {
            predictionModeLabel.textContent = '(Next Word)';
            predictionModeLabel.style.color = '#e67e22';
        }
    }
    
    if (predictionsContainer) {
        if (state.predictions && state.predictions.length > 0) {
            predictionsContainer.innerHTML = state.predictions
                .map((pred, index) => {
                    const highlighted = state.scan_level === 'predictions' && index === state.current_index;
                    return `<div class="prediction-item ${highlighted ? 'highlighted' : ''}">${pred}</div>`;
                })
                .join('');
        } else {
            predictionsContainer.innerHTML = '<div style="color: #999; font-style: italic;">No predictions available</div>';
        }
    }
    
    // Update scanning items
    updateScanningItems(state);
}

/**
 * Update the scanning section based on scan level
 */
function updateScanningItems(state) {
    const scanLabel = document.getElementById('scan-label');
    const scanningContainer = document.getElementById('scanning-items');
    
    // Set appropriate class for scan level styling
    scanningContainer.className = `scanning-items scan-level-${state.scan_level}`;
    
    // Update label based on scan level
    if (state.scan_level === 'keys') {
        scanLabel.textContent = 'Scanning Keys:';
    } else if (state.scan_level === 'letters') {
        scanLabel.textContent = `Scanning Letters on Key ${state.selected_key}:`;
    } else if (state.scan_level === 'predictions') {
        scanLabel.textContent = 'Scanning Predictions:';
    }
    
    // Render scanning items
    if (state.scanning_items && state.scanning_items.length > 0) {
        scanningContainer.innerHTML = state.scanning_items
            .map((item, index) => {
                const highlighted = index === state.current_index;
                return `<div class="scan-item ${highlighted ? 'highlighted' : ''}">${item}</div>`;
            })
            .join('');
    } else {
        scanningContainer.innerHTML = '';
    }
}

/**
 * Handle keyboard input (1, 2, 3 keys)
 */
function handleKeyPress(event) {
    const key = event.key;
    
    if (key === '1') {
        sendState(1);
    } else if (key === '2') {
        sendState(2);
    } else if (key === '3') {
        sendState(3);
    }
}

/**
 * Initialize the application
 */
function init() {
    // Start polling for state updates
    setInterval(fetchState, POLL_INTERVAL);
    
    // Initial fetch
    fetchState();
    
    // Add keyboard listener
    document.addEventListener('keypress', handleKeyPress);
    
    console.log('T9 Keyboard Interface initialized');
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}