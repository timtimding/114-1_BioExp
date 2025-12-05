from flask import Flask, render_template, jsonify, request
import threading
import time
import nltk
from nltk.corpus import brown, words
from nltk.probability import FreqDist
from collections import defaultdict
import re

try:
    from brainlink2classifier import BrainLink2Classifier
    BCI_AVAILABLE = True
except ImportError:
    print("Warning: 'brainlink2classifier.py' not found. BCI features disabled.")
    BCI_AVAILABLE = False

TIMEOUT = 0.7
COOLDOWN = 1.7
PORT = 'COM4'
MODEL_PATH = 'bci_system_v1.pkl'
SCAN_SPEED = 2.0

app = Flask(__name__)

# Global data structures
word_frequencies = {}
bigrams = defaultdict(lambda: defaultdict(int))
ngram_models = {}  # Will store all n-gram levels
trie_root = {}

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        nltk.download('brown', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False

def build_word_frequency_dict():
    """Build word frequency dictionary from Brown corpus"""
    global word_frequencies
    
    print("Building word frequency dictionary...")
    
    # Get all words from Brown corpus
    brown_words = [word.lower() for word in brown.words() if word.isalpha()]
    
    # Calculate frequencies
    freq_dist = FreqDist(brown_words)
    
    # Store top words (limit to reasonable size)
    word_frequencies = {word.upper(): freq for word, freq in freq_dist.most_common(50000)}
    
    print(f"✓ Loaded {len(word_frequencies)} words with frequencies")

def build_trie():
    """Build prefix tree (Trie) for fast word completion"""
    global trie_root
    
    print("Building Trie for fast lookup...")
    
    trie_root = {}
    
    for word in word_frequencies.keys():
        node = trie_root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        # Mark end of word with frequency
        node['$'] = word_frequencies[word]
    
    print("✓ Trie built successfully")

def build_ngrams():
    """Build n-gram models (bigram through 5-gram) for next word prediction with backoff"""
    global bigrams
    
    print("Building n-gram models (2-gram through 5-gram)...")
    
    # Initialize n-gram storage
    unigrams = defaultdict(int)
    bigrams = defaultdict(lambda: defaultdict(int))
    trigrams = defaultdict(lambda: defaultdict(int))
    fourgrams = defaultdict(lambda: defaultdict(int))
    fivegrams = defaultdict(lambda: defaultdict(int))
    
    # Get sentences from Brown corpus
    sentences = brown.sents()
    
    # Build all n-grams
    for sentence in sentences:
        # Clean and normalize
        clean_sent = [word.upper() for word in sentence if word.isalpha()]
        
        if len(clean_sent) < 2:
            continue
        
        # Build unigrams (word frequencies)
        for word in clean_sent:
            unigrams[word] += 1
        
        # Build bigrams
        for i in range(len(clean_sent) - 1):
            w1 = clean_sent[i]
            w2 = clean_sent[i + 1]
            bigrams[w1][w2] += 1
        
        # Build trigrams
        for i in range(len(clean_sent) - 2):
            context = (clean_sent[i], clean_sent[i + 1])
            next_word = clean_sent[i + 2]
            trigrams[context][next_word] += 1
        
        # Build 4-grams
        for i in range(len(clean_sent) - 3):
            context = (clean_sent[i], clean_sent[i + 1], clean_sent[i + 2])
            next_word = clean_sent[i + 3]
            fourgrams[context][next_word] += 1
        
        # Build 5-grams
        for i in range(len(clean_sent) - 4):
            context = (clean_sent[i], clean_sent[i + 1], clean_sent[i + 2], clean_sent[i + 3])
            next_word = clean_sent[i + 4]
            fivegrams[context][next_word] += 1
    
    # Store globally
    global ngram_models
    ngram_models = {
        'unigrams': unigrams,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'fourgrams': fourgrams,
        'fivegrams': fivegrams
    }
    
    # print(f"✓ Built n-gram models:")
    # print(f"  - Unigrams: {len(unigrams)} words")
    # print(f"  - Bigrams: {len(bigrams)} contexts")
    # print(f"  - Trigrams: {len(trigrams)} contexts")
    # print(f"  - 4-grams: {len(fourgrams)} contexts")
    # print(f"  - 5-grams: {len(fivegrams)} contexts")

def get_word_completions(prefix, num_suggestions=5):
    """
    Get word completions using Trie lookup
    Args:
        prefix: partial word like "HEL"
        num_suggestions: number of completions to return
    Returns:
        list of completed words
    """
    if not prefix or not trie_root:
        return []
    
    prefix = prefix.upper()
    
    # Navigate to prefix in Trie
    node = trie_root
    for char in prefix:
        if char not in node:
            return []  # Prefix doesn't exist
        node = node[char]
    
    # Collect all words from this point
    completions = []
    
    def collect_words(current_node, current_word):
        if '$' in current_node:  # End of word
            frequency = current_node['$']
            completions.append((current_word, frequency))
        
        for char, next_node in current_node.items():
            if char != '$':
                collect_words(next_node, current_word + char)
    
    collect_words(node, prefix)
    
    # Sort by frequency (most common first)
    completions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N words
    return [word for word, freq in completions[:num_suggestions]]

def get_next_word_predictions(context, num_suggestions=5):
    """
    Get next word predictions using n-gram model with backoff
    Args:
        context: previous text like "I WANT TO GO"
        num_suggestions: number of predictions
    Returns:
        list of next word predictions
    """
    if not context or not ngram_models:
        return []
    
    # Get words from context
    words = context.strip().upper().split()
    if not words:
        return []
    
    # Try 5-gram first (using last 4 words)
    if len(words) >= 4:
        context_tuple = tuple(words[-4:])
        if context_tuple in ngram_models['fivegrams']:
            predictions = ngram_models['fivegrams'][context_tuple]
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            result = [word for word, count in sorted_preds[:num_suggestions]]
            if len(result) >= num_suggestions:
                print(f"5-gram prediction for {context_tuple}: {result}")
                return result
    
    # Backoff to 4-gram (using last 3 words)
    if len(words) >= 3:
        context_tuple = tuple(words[-3:])
        if context_tuple in ngram_models['fourgrams']:
            predictions = ngram_models['fourgrams'][context_tuple]
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            result = [word for word, count in sorted_preds[:num_suggestions]]
            if len(result) >= num_suggestions:
                print(f"4-gram prediction for {context_tuple}: {result}")
                return result
    
    # Backoff to trigram (using last 2 words)
    if len(words) >= 2:
        context_tuple = tuple(words[-2:])
        if context_tuple in ngram_models['trigrams']:
            predictions = ngram_models['trigrams'][context_tuple]
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            result = [word for word, count in sorted_preds[:num_suggestions]]
            if len(result) >= num_suggestions:
                print(f"Trigram prediction for {context_tuple}: {result}")
                return result
    
    # Backoff to bigram (using last word)
    last_word = words[-1]
    if last_word in ngram_models['bigrams']:
        predictions = ngram_models['bigrams'][last_word]
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        result = [word for word, count in sorted_preds[:num_suggestions]]
        if len(result) >= num_suggestions:
            print(f"Bigram prediction for '{last_word}': {result}")
            return result
    
    # Final backoff to unigrams (most common words overall)
    if ngram_models['unigrams']:
        sorted_unigrams = sorted(ngram_models['unigrams'].items(), key=lambda x: x[1], reverse=True)
        result = [word for word, count in sorted_unigrams[:num_suggestions]]
        print(f"Unigram fallback (most common words): {result}")
        return result
    
    return []


class T9KeyboardWithDictionary:
    def __init__(self, scan_speed=0.8):
        """
        T9 Keyboard with dictionary-based predictions
        scan_speed: seconds between each scan step
        """
        self.scan_speed = scan_speed
        
        # T9 key mappings
        self.keys = {
            '1': [' ', '?', '!'],
            '2': ['A', 'B', 'C'],
            '3': ['D', 'E', 'F'],
            '4': ['G', 'H', 'I'],
            '5': ['J', 'K', 'L'],
            '6': ['M', 'N', 'O'],
            '7': ['P', 'Q', 'R', 'S'],
            '8': ['T', 'U', 'V'],
            '9': ['W', 'X', 'Y', 'Z']
        }
        
        self.key_list = list(self.keys.keys())
        
        # State tracking
        self.scan_level = 'keys'
        self.current_index = 0
        self.selected_key = None
        self.output_text = ""
        self.current_word = ""
        
        # Word prediction
        self.current_predictions = []
        self.prediction_mode = 'completion'
        
        # Threading
        self.lock = threading.Lock()
        self.scanning = False
        
    def _get_predictions(self, prefix_or_context):
        """Get predictions using dictionary or bigrams"""
        if self.prediction_mode == 'completion':
            # Word completion mode - use Trie
            results = get_word_completions(prefix_or_context, num_suggestions=5)
            print(f"Completions for '{prefix_or_context}': {results}")
            return results
        else:
            # Next word prediction mode - use bigrams
            results = get_next_word_predictions(prefix_or_context, num_suggestions=5)
            print(f"Next word predictions for '{prefix_or_context}': {results}")
            return results
    
    def scan_next(self):
        """Move to next item in current scan level"""
        with self.lock:
            if self.scan_level == 'keys':
                self.current_index = (self.current_index + 1) % len(self.key_list)
            elif self.scan_level == 'letters':
                self.current_index = (self.current_index + 1) % len(self.keys[self.selected_key])
            elif self.scan_level == 'predictions':
                self.current_index = (self.current_index + 1) % len(self.current_predictions)
    
    def handle_state_1(self):
        """State 1: Cancel/return to key scanning"""
        with self.lock:
            # Cancel from any level - always return to key scanning
            if self.scan_level in ['letters', 'predictions']:
                self.scan_level = 'keys'
                self.current_index = 0
                self.selected_key = None
    
    def handle_state_2(self):
        """State 2: Confirm selection"""
        with self.lock:
            if self.scan_level == 'keys':
                # Selected a key, now scan its letters
                self.selected_key = self.key_list[self.current_index]
                self.scan_level = 'letters'
                self.current_index = 0
                
            elif self.scan_level == 'letters':
                # Selected a letter, add to output
                selected_letter = self.keys[self.selected_key][self.current_index]
                self.output_text += selected_letter
                
                if selected_letter == ' ':
                    # Space pressed - switch to next word prediction
                    self.current_word = ""
                    self.prediction_mode = 'next_word'
                    # Get next word predictions based on context
                    context = self.output_text.strip()
                    if context:
                        self.current_predictions = self._get_predictions(context)
                    else:
                        self.current_predictions = []
                else:
                    # Letter pressed - word completion mode
                    self.current_word += selected_letter
                    self.prediction_mode = 'completion'
                    self.current_predictions = self._get_predictions(self.current_word)
                
                # Reset to key scanning
                self.scan_level = 'keys'
                self.current_index = 0
                self.selected_key = None
                
            elif self.scan_level == 'predictions':
                # Confirmed a prediction
                selected_prediction = self.current_predictions[self.current_index]
                
                if self.prediction_mode == 'completion':
                    # Replace current word with prediction
                    self.output_text = self.output_text[:-len(self.current_word)] + selected_prediction + " "
                    self.current_word = ""
                    # Switch to next word mode
                    self.prediction_mode = 'next_word'
                    context = self.output_text.strip()
                    if context:
                        self.current_predictions = self._get_predictions(context)
                    else:
                        self.current_predictions = []
                else:
                    # Next word prediction - just append
                    self.output_text += selected_prediction + " "
                    self.current_word = ""
                    # Stay in next word mode
                    context = self.output_text.strip()
                    self.current_predictions = self._get_predictions(context)
                
                # Reset to key scanning
                self.scan_level = 'keys'
                self.current_index = 0
                self.selected_key = None
    
    def handle_state_3(self):
        """State 3: Enter prediction mode"""
        with self.lock:
            if self.current_predictions and self.scan_level != 'predictions':
                self.scan_level = 'predictions'
                self.current_index = 0
    
    def get_state(self):
        """Get current keyboard state for frontend"""
        with self.lock:
            state = {
                'output_text': self.output_text,
                'current_word': self.current_word,
                'predictions': self.current_predictions,
                'prediction_mode': self.prediction_mode,
                'scan_level': self.scan_level,
                'current_index': self.current_index,
            }
            
            # Add the items being scanned
            if self.scan_level == 'keys':
                # Show keys with letters
                state['scanning_items'] = [
                    f"{key}: {'/'.join(self.keys[key])}" for key in self.key_list
                ]
            elif self.scan_level == 'letters':
                state['scanning_items'] = self.keys[self.selected_key]
                state['selected_key'] = self.selected_key
            elif self.scan_level == 'predictions':
                state['scanning_items'] = self.current_predictions
            
            return state


# Initialize NLTK data at startup
print("=" * 60)
print("Initializing T9 Keyboard with Dictionary & N-grams...")
print("=" * 60)

data_loaded = download_nltk_data()

if data_loaded:
    build_word_frequency_dict()
    build_trie()
    build_ngrams()
    print("✓ Dictionary and N-gram model ready!")
else:
    print("✗ Failed to load NLTK data")

# Initialize keyboard
keyboard = T9KeyboardWithDictionary(scan_speed=SCAN_SPEED)

def bci_input_handler(is_focus, blink_count):
    """
    這個函式會由 BCI Driver 在後台自動呼叫。
    它直接操作全域變數 'keyboard'，不需透過 HTTP Request。
    """
    # Change scan_speed according to state(relaxed/focus)
    if is_focus:
        if keyboard.scan_speed != SCAN_SPEED:
            print("Focus detected! Resuming scan.")
            keyboard.scan_speed = SCAN_SPEED
    else:
        if keyboard.scan_speed != 999:
            
            print("Relax detected! Pausing scan.")
            keyboard.scan_speed = 999

    if blink_count > 0:
        print(f"[BCI Command] Blink Count: {blink_count}")
        
        # 對應原本 API /api/input/<state> 的邏輯
        if is_focus and blink_count == 1:
            print(" -> Trigger: State 1 (Cancel/Return)")
            keyboard.handle_state_1()
            
        elif is_focus and blink_count == 2:
            print(" -> Trigger: State 2 (Select/Confirm)")
            keyboard.handle_state_2()
            
        elif is_focus and blink_count == 3:
            print(" -> Trigger: State 3 (Prediction Mode)")
            keyboard.handle_state_3()
# ==================================

def auto_scan():
    """Auto-scanning background thread"""
    keyboard.scanning = True

    while keyboard.scanning:
        # Pause scanning if the state is "relaxed"
        if keyboard.scan_speed >= 999:
            time.sleep(0.1)
            continue
        time.sleep(keyboard.scan_speed)
        if keyboard.scan_speed < 999:
            keyboard.scan_next()

# Start scanning thread
scan_thread = threading.Thread(target=auto_scan, daemon=True)
scan_thread.start()


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """API endpoint to get current keyboard state"""
    return jsonify(keyboard.get_state())

@app.route('/api/input/<int:state>', methods=['POST'])
def send_input(state):
    """API endpoint to send state input (1, 2, or 3)"""
    if state == 1:
        keyboard.handle_state_1()
    elif state == 2:
        keyboard.handle_state_2()
    elif state == 3:
        keyboard.handle_state_3()
    else:
        return jsonify({'error': 'Invalid state'}), 400
    
    return jsonify({'success': True})


if __name__ == '__main__':
    if data_loaded:
        print("=" * 60)
        print("✓ T9 Keyboard Server with 5-gram Model Ready!")
        print("=" * 60)
        print("Open your browser to: http://localhost:5000")
        print("=" * 60)
        print("\nFeatures:")
        print("  • Instant word completion (Trie-based)")
        print("  • Smart next word prediction (5-gram with backoff)")
        print("  • 50,000 word dictionary from NLTK Brown corpus")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Server starting WITHOUT dictionary (NLTK load failed)")
        print("=" * 60)
    
    bci_driver = None
    if BCI_AVAILABLE:
        print("Starting BrainLink BCI Driver...")

        bci_driver = BrainLink2Classifier(
            port=PORT,
            baud=57600,
            timeout=TIMEOUT,
            cooldown=COOLDOWN,
            model_path=MODEL_PATH
        )
        
        # 設定我們剛剛寫好的處理函式
        bci_driver.set_callback(bci_input_handler)
        
        # 啟動監聽
        bci_driver.start()
        print("✓ BCI Connected: Blink 1=Cancel, 2=Select, 3=Predict")
        print("=" * 60)
    # ==================================
    
    try:
        # 注意：use_reloader=False 是必須的，避免 Serial Port 衝突
        app.run(debug=True, use_reloader=False)
    finally:
        # 當 Flask 結束時，安全關閉藍芽連線
        if bci_driver:
            bci_driver.stop()