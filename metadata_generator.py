import csv
import logging
import requests
import langdetect
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator
import backoff
from pathlib import Path
import sys
import json
from tqdm import tqdm
import os
import time
import random

class RateLimiter:
    def __init__(self, min_delay: int = 1, max_delay: int = 5):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0

    def wait(self) -> None:
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_delay:
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
        self.last_request_time = time.time()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_name: str
    review_model_name: str
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None, model_name: Optional[str] = None) -> 'Config':
        default_config = {
            "model_name": "google/gemini-2.0-flash-001",
            "review_model_name": "anthropic/claude-3.7-sonnet"
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    default_config.update(config_data)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        # Override model_name if provided
        if model_name:
            default_config["model_name"] = model_name
            logger.info(f"Using model: {model_name}")
        
        return cls(**default_config)

class ContentProcessor:
    def __init__(self, text: str):
        self.text = text.strip()
    
    def is_valid(self) -> bool:
        return bool(self.text) and len(self.text) >= 50

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            return 'en'

class MetadataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limiter = RateLimiter(min_delay=3, max_delay=5)  # Add delay between API calls

    @staticmethod
    def truncate_summary(summary: str, max_length: int = 250) -> str:
        if len(summary) > max_length:
            last_full_stop = summary.rfind('.', 0, max_length)
            return summary[:last_full_stop + 1] if last_full_stop != -1 else summary[:max_length]
        return summary

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        on_backoff=lambda details: logger.warning(f"Retrying API call: {details}")
    )
    def _make_completion_call(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        model_name: str = None
    ) -> str:
        self.rate_limiter.wait()  # Add rate limiting between API calls
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name or self.config.model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            return response_data["choices"][0]["message"]["content"].strip() if response_data.get("choices") else ""
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def summarize_content(self, content: str) -> str:
        processor = ContentProcessor(content)
        if not processor.is_valid():
            return "Content too short or invalid for summarization"

        language = processor.detect_language(content)
        
        prompt_text = (
            "As a search engine optimization expert, using natural language processing, provide a concise, complete summary suitable "
            "for a meta description in English. The summary should be informative, use topic-specific terms, in full sentences, "
            "and must end concisely within 275 characters. Avoid using ellipses or cutting off sentences.\n\n"
            f"{content}\n\nSummary:"
        ) if language == 'en' else (
            "À l'aide du traitement automatique du langage naturel, fournissez un résumé concis et complet adapté "
            "à une méta-description en français. Le résumé doit être informatif, en phrases complètes, "
            "et doit se terminer de manière concise dans les 275 caractères. Évitez l'utilisation de points de suspension ou de coupures abruptes.\n\n"
            f"{content}\n\nRésumé:"
        )
        
        summary = self._make_completion_call(
            prompt_text=prompt_text,
            max_tokens=200,
            temperature=0.5,
            model_name=self.config.model_name
        )
        return self.truncate_summary(summary)

    def generate_keywords(self, content: str) -> str:
        processor = ContentProcessor(content)
        if not processor.is_valid():
            return "Content too short or invalid for keyword generation"

        prompt_text = (
            "As a search engine optimization expert, identify and extract 10 meaningful, topic-specific meta keywords from the following content. "
            "Please list the keywords in a comma-separated format only. Do not include any additional notes, explanations, or commentary. "
            "Exclude 'Canada Revenue Agency' from the keywords. "
            "Focus strictly on providing the keywords.:\n\n"
            f"{content}\n\nKeywords:"
        )
        
        return self._make_completion_call(
            prompt_text=prompt_text,
            max_tokens=80,
            temperature=0.3,
            model_name=self.config.model_name
        )

    def translate_to_french(self, text: str, is_description: bool = True) -> str:
        if not text or text.strip() == "":
            return ""
            
        if is_description:
            prompt_text = (
                "Translate the following English meta description to French. IMPORTANT: Your response must contain ONLY the direct translation, "
                "with absolutely NO commentary, NO suggestions, NO explanations, and NO additional text of any kind. "
                "Return ONLY the translated text itself:\n\n"
                f"{text}\n\n"
                "French translation:"
            )
            
            try:
                translated = self._make_completion_call(
                    prompt_text=prompt_text,
                    max_tokens=200,
                    temperature=0.3,
                    model_name=self.config.model_name
                )
                
                # Post-processing to remove any commentary
                if ":" in translated:
                    # If there's a colon, it might be separating commentary from translation
                    parts = translated.split(":", 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 10:  # Ensure we're not just removing part of the translation
                        translated = parts[1].strip()
                
                # Remove common commentary phrases
                commentary_phrases = [
                    "Voici la traduction", "La traduction est", "Traduction:",
                    "En français:", "Je traduis:", "Traduction française:"
                ]
                for phrase in commentary_phrases:
                    if translated.startswith(phrase):
                        translated = translated[len(phrase):].strip()
                
                return translated
            except Exception as e:
                logger.error(f"Error translating description: {str(e)}")
                return "Failed to translate description. Please try again with a different model."
        else:
            # For keywords, ensure comma-delimited format
            prompt_text = (
                "Translate each of these English keywords to French. IMPORTANT: Return ONLY the translated keywords "
                "in a comma-separated list. Provide absolutely NO commentary, NO suggestions, NO explanations, and NO additional text of any kind. "
                "Return ONLY a comma-separated list of the translated keywords:\n\n"
                f"{text}\n\n"
                "French keywords (comma-separated):"
            )
            
            try:
                translated = self._make_completion_call(
                    prompt_text=prompt_text,
                    max_tokens=80,
                    temperature=0.3,
                    model_name=self.config.model_name
                )
                
                # Post-processing to remove any commentary
                if ":" in translated:
                    # If there's a colon, it might be separating commentary from translation
                    parts = translated.split(":", 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 10:  # Ensure we're not just removing part of the translation
                        translated = parts[1].strip()
                
                # Remove common commentary phrases
                commentary_phrases = [
                    "Voici les mots-clés", "Les mots-clés sont", "Mots-clés:",
                    "En français:", "Je traduis:", "Traduction française:"
                ]
                for phrase in commentary_phrases:
                    if translated.startswith(phrase):
                        translated = translated[len(phrase):].strip()
                
                # Clean up the response to ensure proper comma-delimited format
                keywords = [k.strip() for k in translated.split(',')]
                return ', '.join(k for k in keywords if k)
            except Exception as e:
                logger.error(f"Error translating keywords: {str(e)}")
                return "Failed to translate keywords. Please try again with a different model."
    
    def review_french_meta_description(self, french_content: str, translated_desc: str) -> str:
        prompt_text = (
            "As a bilingual SEO expert fluent in French, review the following translated meta description against the original French content. "
            "Analyze for:\n"
            "1. Translation accuracy and naturalness\n"
            "2. SEO effectiveness in French\n"
            "3. Cultural and linguistic appropriateness\n"
            "4. Consistency with the original content\n\n"
            "Provide specific suggestions for improvements if needed.\n\n"
            f"French Content: {french_content}\n\n"
            f"Translated Meta Description: {translated_desc}\n\n"
            "Analysis and Suggestions:"
        )
        
        try:
            # Always use Claude Sonnet for French review functionality
            review = self._make_completion_call(
                prompt_text=prompt_text,
                max_tokens=400,
                temperature=0.3,
                model_name="anthropic/claude-3.7-sonnet"
            )
            
            return review
        except Exception as e:
            logger.error(f"Error reviewing description: {str(e)}")
            return "Failed to review description. Please try again with a different model."
        
    def review_french_keywords(self, french_content: str, translated_keywords: str) -> str:
        prompt_text = (
            "As a bilingual SEO expert fluent in French, review the following translated keywords against the original French content. "
            "Analyze for:\n"
            "1. Translation accuracy and relevance\n"
            "2. SEO effectiveness for French market\n"
            "3. Cultural and market appropriateness\n"
            "4. Consistency with French industry terminology\n\n"
            "Provide specific suggestions for improvements if needed.\n\n"
            f"French Content: {french_content}\n\n"
            f"Translated Keywords: {translated_keywords}\n\n"
            "Analysis and Suggestions:"
        )
        
        try:
            # Always use Claude Sonnet for French review functionality
            review = self._make_completion_call(
                prompt_text=prompt_text,
                max_tokens=400,
                temperature=0.3,
                model_name="anthropic/claude-3.7-sonnet"
            )
            
            return review
        except Exception as e:
            logger.error(f"Error reviewing keywords: {str(e)}")
            return "Failed to review keywords. Please try again with a different model."
    
    def review_metadata(self, content: str, description: str, keywords: str) -> str:
        """
        Evaluate the SEO effectiveness of generated metadata.
        
        This method analyzes the provided content, description, and keywords to evaluate
        SEO effectiveness. It automatically detects the language of the content and applies
        the appropriate language-specific prompt for evaluation.
        
        Note: The calling function must properly initialize variables to store the return value
        before calling this method to prevent reference errors.
        """
        logger.info(f"Starting metadata review with model: {self.config.review_model_name}")
        logger.info(f"Input Description: {description}")
        logger.info(f"Input Keywords: {keywords}")
        
        # Detect language of the content
        language = ContentProcessor.detect_language(content)
        
        # Create appropriate prompt based on language
        if language == 'fr':
            prompt_text = (
                "Vous êtes un expert en optimisation pour les moteurs de recherche (SEO) avec une connaissance approfondie de l'optimisation des métadonnées. "
                "Examinez le contenu suivant et ses métadonnées en vous concentrant sur la maximisation de la visibilité dans les moteurs de recherche et l'engagement des utilisateurs. "
                "Analysez la méta-description et les mots-clés pour:\n"
                "1. Pertinence par rapport au contenu\n"
                "2. Efficacité de l'optimisation pour les moteurs de recherche\n"
                "3. Densité et placement des mots-clés\n"
                "4. Potentiel d'engagement des utilisateurs\n\n"
                "Fournissez des recommandations spécifiques et exploitables pour les améliorations. Si les métadonnées sont déjà optimales, expliquez pourquoi.\n\n"
                f"Contenu: {content}\n\n"
                f"Méta-description actuelle: {description}\n\n"
                f"Mots-clés actuels: {keywords}\n\n"
                "Analyse SEO experte et recommandations:"
            )
        else:  # Default to English
            prompt_text = (
                "You are an expert Search Engine Optimization (SEO) specialist with extensive knowledge in metadata optimization. "
                "Review the following content and its metadata with a focus on maximizing search engine visibility and user engagement. "
                "Analyze the meta description and keywords for:\n"
                "1. Relevance to content\n"
                "2. Search engine optimization effectiveness\n"
                "3. Keyword density and placement\n"
                "4. User engagement potential\n\n"
                "Provide specific, actionable recommendations for improvements. If the metadata is already optimal, explain why.\n\n"
                f"Content: {content}\n\n"
                f"Current Meta Description: {description}\n\n"
                f"Current Keywords: {keywords}\n\n"
                "Expert SEO Analysis and Recommendations:"
            )
        
        try:
            recommendations = self._make_completion_call(
                prompt_text=prompt_text,
                max_tokens=400,
                temperature=0.3,
                model_name=self.config.review_model_name
            )
            
            logger.info("Received recommendations from model")
            logger.info(f"Recommendations: {recommendations}")
            
            if not recommendations or len(recommendations.strip()) < 10:
                logger.error("Received empty or very short recommendations")
                return "Error: Model provided insufficient recommendations"
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error during metadata review: {str(e)}")
            return f"Error during review: {str(e)}"

class CSVHandler:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def validate_files(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not self.input_path.suffix == '.csv':
            raise ValueError("Input file must be a CSV file")

    def read_csv(self) -> Iterator[Dict]:
        try:
            with open(self.input_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                yield from reader
        except UnicodeDecodeError:
            with open(self.input_path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                yield from reader

    def write_row(self, writer: csv.DictWriter, row: Dict) -> None:
        try:
            writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing row: {e}")
            raise

def main():
    try:
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.error("OPENROUTER_API_KEY environment variable is not set")
            sys.exit(1)
            
        config = Config.load_config()
        processor = MetadataProcessor(config)
        csv_handler = CSVHandler('scraped_content.csv', 'processed_metadata.csv')
        
        csv_handler.validate_files()
        
        total_rows = sum(1 for _ in csv_handler.read_csv())
        
        with open(csv_handler.output_path, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['url', 'scraped_content', 'generated_description', 'generated_keywords', 'model_recommendations']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            with tqdm(total=total_rows, desc="Processing content") as pbar:
                for row in csv_handler.read_csv():
                    try:
                        scraped_content = row['scraped_content']
                        
                        description = processor.summarize_content(scraped_content)
                        keywords = processor.generate_keywords(scraped_content)
                        
                        recommendations = processor.review_metadata(scraped_content, description, keywords)
                        
                        output_row = {
                            'url': row['url'],
                            'scraped_content': scraped_content,
                            'generated_description': description,
                            'generated_keywords': keywords,
                            'model_recommendations': recommendations
                        }
                        
                        csv_handler.write_row(writer, output_row)
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        continue

        logger.info(f'Metadata generation completed. Results saved to {csv_handler.output_path}')
        sys.exit(0)  # Explicitly exit after completion
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
