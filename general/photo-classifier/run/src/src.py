import os
import json
from io import BytesIO
import base64
import logging
import numpy as np
import cv2
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description='PDF Image Processing and Classification')
    
    # Add arguments for all configurable parameters
    parser.add_argument('--large-image-threshold', 
                      type=float,
                      default=3.7 * 1024 * 1024,
                      help='Threshold for large image detection in pixels (default: 3.7MB)')
    
    parser.add_argument('--target-width',
                      type=int,
                      default=800,
                      help='Target image width for processing (default: 800)')
    
    parser.add_argument('--target-height',
                      type=int,
                      default=800,
                      help='Target image height for processing (default: 800)')
    
    parser.add_argument('--max-file-size',
                      type=int,
                      default=2 * 1024 * 1024,
                      help='Maximum file size in bytes (default: 2MB)')
    
    parser.add_argument('--input-dir',
                      type=str,
                      default='data/input',
                      help='Input directory for PDF files')
    
    parser.add_argument('--output-dir',
                      type=str,
                      default='data/output',
                      help='Output directory for results')
    
    parser.add_argument('--model',
                      type=str,
                      default='claude-3-haiku-20240307',
                      help='Claude model to use for classification')
    
    parser.add_argument('--temperature',
                      type=float,
                      default=0.0,
                      help='Temperature for Claude model')
    
    return parser.parse_args()

class PhotoClassifier:
    def __init__(self, args):
        self.large_image_threshold = args.large_image_threshold
        self.target_image_size = (args.target_width, args.target_height)
        self.max_file_size = args.max_file_size
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        
        self.chat = ChatAnthropic(
            model=args.model,
            temperature=args.temperature
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


    def preprocess_image(image):
        """Apply advanced preprocessing techniques to enhance overall image quality"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Convert to RGB if image is in RGBA mode
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        # Convert to LAB color space for more accurate color processing
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel to improve overall contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge back and convert to RGB
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Apply bilateral filter to smooth while preserving edges
        smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)

        # Adjust gamma to enhance details in darker regions
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        gamma_corrected = cv2.LUT(smooth, table)

        # Convert back to PIL Image
        processed_image = Image.fromarray(gamma_corrected)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        contrast_enhanced = enhancer.enhance(1.2)

        # Enhance color
        color_enhancer = ImageEnhance.Color(contrast_enhanced)
        color_enhanced = color_enhancer.enhance(1.1)

        # Enhance sharpness one more time
        sharpness_enhancer = ImageEnhance.Sharpness(color_enhanced)
        final_image = sharpness_enhancer.enhance(1.3)

        return final_image

   
    def resize_image(self, image):
        """Resize the image to fit within the target size while maintaining aspect ratio"""
        image.thumbnail(self.target_image_size, Image.Resampling.LANCZOS)
        return image

    def encode_image_with_size_control(self, image, initial_quality=85):
        """Encode an image to base64 with size control"""
        quality = initial_quality
        while quality > 20:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
            img_str = base64.b64encode(buffered.getvalue())
            if len(img_str) <= self.max_file_size:
                return img_str.decode("utf-8")
            quality -= 5
        raise ValueError(
            "Unable to compress image to required size while maintaining acceptable quality"
        )


    def get_page_description(base64_image, self):
        """Get a description of the page using Claude Haiku"""

        prompt = """
        <task_description>
        Analyze the input image and determine whether it is a photograph or not. This is a binary classification task. Return True if the input is a photograph, and False otherwise. Do not provide any additional explanation or commentary.
        </task_description>

        <context>
        This classification will be used to quickly sort inputs into photographs and non-photographs. The majority of inputs may be documents or other non-photographic images, but the focus is on accurately identifying true photographs when they occur.
        </context>

        <classification_categories>
        1. Photograph: True
        2. Non-Photograph: False
        </classification_categories>

        <thinking_process>
        1. Observe the visual characteristics of the input.
        2. Identify key features that suggest whether it's a photograph or not.
        3. Consider any ambiguities or edge cases.
        4. Make a final determination based on the overall assessment.
        5. Return only True or False based on this determination.
        </thinking_process>

        <classification_guidelines>

        <photograph_indicators>
        - Realistic representation of real-world scenes or objects
        - Natural lighting, shadows, and textures
        - Depth of field and focus typical of camera lenses
        - Presence of photographic artifacts (e.g., lens flare, motion blur)
        </photograph_indicators>

        <non_photograph_indicators>
        - Stylized or abstract representations
        - Predominantly text-based content (e.g., documents, screenshots of text)
        - Graphical elements like charts, diagrams, or user interface components
        - Hand-drawn or digitally created illustrations
        - Computer-generated imagery or renderings
        </non_photograph_indicators>

        </classification_guidelines>

        <additional_instructions>
        - If the input is ambiguous, classify it based on the predominant characteristics.
        - Do not provide any explanation or reasoning in the output.
        - If no image is provided or there are technical issues preventing analysis, return False.
        </additional_instructions>

        <output_format>
        [True/False]
        </output_format>
        """

        try:
            msg = self.chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    )
                ]
            )
            print(msg)
            return msg.content.strip()
        except Exception as e:
            logging.error(f"Error getting page description: {str(e)}")
            return f"Error occurred while processing the image: {str(e)}"



    def process_page(self, pdf_path, page_number):
        """Process a single page of a PDF"""
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                images = convert_from_path(
                    pdf_path, first_page=page_number + 1, last_page=page_number + 1
                )
                
                if any(issubclass(warn.category, Image.DecompressionBombWarning) for warn in w):
                    logging.warning(f"DecompressionBombWarning for page {page_number+1} of {pdf_path}")
                    return page_number + 1, True
            
            image = images[0]
            
            if image.width * image.height > self.large_image_threshold:
                logging.info(f"Large image detected on page {page_number+1} of {pdf_path}")
                return page_number + 1, True

            resized_image = self.resize_image(image)
            preprocessed_image = self.preprocess_image(resized_image)
            base64_image = self.encode_image_with_size_control(preprocessed_image)
            is_photo = self.get_page_description(base64_image, self)

            logging.info(f"Processed page {page_number+1} of {pdf_path}")
            return page_number + 1, is_photo
            
        except Exception as e:
            logging.warning(
                f"Error processing page {page_number+1} of {pdf_path}: {str(e)}"
            )
            return page_number + 1, False

    def process_pdf(self, pdf_path):
        """Process a PDF file page by page in parallel"""
        results = {}

        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            page_count = len(reader.pages)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_page, pdf_path, page_number)
                for page_number in range(page_count)
            ]
            for future in as_completed(futures):
                page_number, is_photo = future.result()
                results[page_number] = is_photo

        return results
    
def main():
    args = parse_args()
    
    load_dotenv(find_dotenv())
    
    processor = PhotoClassifier(args)
    
    # Process all PDFs in input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(args.input_dir, filename)
            logging.info(f"Processing {filename}")
            
            try:
                results = processor.process_pdf(pdf_path)
                
                # Save results
                output_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(filename)[0]}_results.json"
                )
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)
                    
                logging.info(f"Results saved to {output_path}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()