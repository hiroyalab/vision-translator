
import os
import re
import sys
import subprocess
import webview
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator
import objc
from Foundation import NSURL
from Vision import (
    VNImageRequestHandler,
    VNRecognizeTextRequest,
    VNRequestTextRecognitionLevelAccurate
)
from PIL import Image, ImageDraw, ImageFont

class API:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=10) # Increased workers for translation
        self.temp_image = "temp_capture.png"
        self._cached_font_path = self._find_font()
        # Cleanup any leftover from previous runs
        self._cleanup()

    def _cleanup(self):
        if os.path.exists(self.temp_image):
            try:
                os.remove(self.temp_image)
            except:
                pass

    def _find_font(self):
        font_paths = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Cache/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf"
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                return fp
        return None

    def _translate_fragment(self, text):
        try:
            # Skip translation if text already contains Japanese characters
            if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
                return text
            return GoogleTranslator(source='auto', target='ja').translate(text)
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            return text

    def capture_and_translate(self):
        try:
            # 1. Capture screen region
            subprocess.run(["screencapture", "-i", self.temp_image], check=True)
            
            if not os.path.exists(self.temp_image):
                return {"error": "Canceled or failed to capture"}

            # 2. Perform OCR
            results = self._ocr_vision(self.temp_image)
            print(f"OCR found {len(results)} fragments")
            
            if not results:
                return {"error": "No text detected"}

            # 3. Parallel Translation
            texts_to_translate = [item['text'] for item in results]
            translated_texts = list(self._executor.map(self._translate_fragment, texts_to_translate))

            # 4. Process image
            import io
            with Image.open(self.temp_image) as img:
                draw = ImageDraw.Draw(img)
                width, height = img.size
                
                # Load font from cache
                if self._cached_font_path:
                    font = ImageFont.truetype(self._cached_font_path, 14)
                else:
                    font = ImageFont.load_default()

                for i, item in enumerate(results):
                    translated = translated_texts[i]
                    box = item['box']
                    
                    x = box['x'] * width
                    y = (1.0 - (box['y'] + box['h'])) * height
                    w = box['w'] * width
                    h = box['h'] * height

                    # Draw background and text
                    padding = 2
                    draw.rectangle(
                        [x - padding, y - padding, x + w + padding, y + h + padding],
                        fill="white"
                    )
                    draw.text((x, y), translated, fill="black", font=font)

                # Save to buffer instead of file to speed up
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_data = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Cleanup
            if os.path.exists(self.temp_image):
                os.remove(self.temp_image)

            return {
                "image": f"data:image/png;base64,{img_data}"
            }
        except subprocess.CalledProcessError:
            return {"error": "Capture cancelled"}
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}

    def _ocr_vision(self, image_path):
        """Native macOS OCR using Vision Framework"""
        input_url = NSURL.fileURLWithPath_(os.path.abspath(image_path))
        handler = VNImageRequestHandler.alloc().initWithURL_options_(input_url, None)
        
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        
        # Explicitly set languages to Japanese and English
        # 'ja-JP' is the code for Japanese.
        try:
            request.setRecognitionLanguages_(['ja-JP', 'en-US'])
        except Exception as e:
            print(f"Warning: Could not set recognition languages: {e}")

        error = None
        success = handler.performRequests_error_([request], error)
        if not success:
            return []

        results = request.results()
        data = []
        for result in results:
            candidates = result.topCandidates_(1)
            if candidates:
                text = candidates[0].string()
                rect = result.boundingBox()
                # Vision coordinates: 0,0 is bottom-left
                data.append({
                    "text": text,
                    "box": {
                        "x": rect.origin.x,
                        "y": rect.origin.y,
                        "w": rect.size.width,
                        "h": rect.size.height
                    }
                })
        
        return data

def main():
    api = API()
    ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'index.html')
    
    window = webview.create_window(
        'Vision Translator', 
        ui_path, 
        js_api=api,
        width=1000,
        height=800,
        transparent=False,
        background_color='#0f172a'
    )
    
    webview.start()

if __name__ == '__main__':
    main()
