import streamlit as st
import pandas as pd
from scrape_urls import URLScraper, RateLimiter
from metadata_generator import Config, MetadataProcessor, ContentProcessor
import tempfile
import os
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input_csv(df, input_type):
    if input_type == "urls_only":
        if len(df.columns) < 1:
            st.error("CSV file must contain at least one column with URLs")
            return False
        return True
    elif input_type == "english_french_pairs":
        required_columns = ['english_url', 'french_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"CSV file must contain the following columns: {', '.join(missing_columns)}")
            return False
        return True
    else:  # urls_and_content
        required_columns = ['url', 'scraped_content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"CSV file must contain the following columns: {', '.join(missing_columns)}")
            return False
        return True

def process_urls(uploaded_file, progress_bar, status_text, input_type):
    # Check if uploaded_file is already a DataFrame
    if isinstance(uploaded_file, pd.DataFrame):
        df = uploaded_file
    else:
        # Otherwise, assume it's a file-like object and read it
        df = pd.read_csv(uploaded_file)
    
    if not validate_input_csv(df, input_type):
        return None
        
    if input_type == "urls_only":
        scraper = URLScraper()
        total_urls = len(df)
        output_data = []
        first_column = df.columns[0]
        
        for i, row in df.iterrows():
            url = row[first_column].strip()
            status_text.text(f'Processing URL {i+1}/{total_urls}: {url}')
            
            scraped_content = scraper.scrape_url(url)
            
            # Determine if the URL is French or English based on '/fr/' in the path
            is_french = '/fr/' in url.lower()
            
            output_data.append({
                'English url': url if not is_french else '',
                'English scraped content': scraped_content if not is_french else '',
                'French url': url if is_french else '',
                'French scraped content': scraped_content if is_french else ''
            })
            
            progress_bar.progress((i + 1) / total_urls)
        
        return pd.DataFrame(output_data)
    elif input_type == "english_french_pairs":
        scraper = URLScraper()
        total_pairs = len(df)
        output_data = []
        
        for i, row in df.iterrows():
            en_url = row['english_url'].strip()
            fr_url = row['french_url'].strip()
            
            if not fr_url or '/fr/' not in fr_url.lower():
                st.warning(f"Skipping pair - French URL must contain '/fr/': {fr_url}")
                continue
                
            status_text.text(f'Processing URL pair {i+1}/{total_pairs}')
            
            # Scrape English content
            en_content = scraper.scrape_url(en_url) if en_url else ''
            # Scrape French content
            fr_content = scraper.scrape_url(fr_url) if fr_url else ''
            
            output_data.append({
                'English url': en_url,
                'English scraped content': en_content,
                'French url': fr_url,
                'French scraped content': fr_content
            })
            
            progress_bar.progress((i + 1) / total_pairs)
        
        return pd.DataFrame(output_data)
    else:  # urls_and_content
        # Rename columns if needed to match expected format
        df = df.rename(columns={
            df.columns[0]: 'url' if 'url' not in df.columns else df.columns[0],
            df.columns[1]: 'scraped_content' if 'scraped_content' not in df.columns else df.columns[1]
        })
        progress_bar.progress(1.0)
        status_text.text('Content already provided, proceeding to metadata generation')
        
        # Convert to the new format for compatibility
        output_data = []
        for i, row in df.iterrows():
            url = row['url'].strip()
            scraped_content = row['scraped_content']
            
            # Determine if the URL is French or English based on '/fr/' in the path
            is_french = '/fr/' in url.lower()
            
            output_data.append({
                'English url': url if not is_french else '',
                'English scraped content': scraped_content if not is_french else '',
                'French url': url if is_french else '',
                'French scraped content': scraped_content if is_french else ''
            })
        
        return pd.DataFrame(output_data)

def generate_metadata(scraped_df, progress_bar, status_text, model_status, input_type="urls_only", model_choice="google/gemini-2.0-flash-001"):
    try:
        config = Config.load_config(model_name=model_choice)
        processor = MetadataProcessor(config)
        
        total_rows = len(scraped_df)
        results = []
        
        for i, row in scraped_df.iterrows():
            try:
                model_status.text(f'Model being used: {config.model_name}')
                
                # Initialize SEO evaluation variables - critical to prevent variable reference errors
                # This initialization ensures proper variable scope and prevents errors when processing different input types
                en_seo_evaluation = ""
                fr_seo_evaluation = ""
                
                # Generate English metadata if we have English content
                en_description = ""
                en_keywords = ""
                if row['English scraped content']:
                    status_text.text(f'Generating English metadata for URL {i+1}/{total_rows}')
                    en_description = processor.summarize_content(row['English scraped content'])
                    en_keywords = processor.generate_keywords(row['English scraped content'])
                    # Perform SEO effectiveness evaluation for English content
                    status_text.text(f'Evaluating English SEO for URL {i+1}/{total_rows}')
                    en_seo_evaluation = processor.review_metadata(row['English scraped content'], en_description, en_keywords)
                
                # Generate French metadata if we have French content
                fr_description = ""
                fr_keywords = ""
                if row.get('French scraped content'):
                    status_text.text(f'Generating French metadata for URL {i+1}/{total_rows}')
                    fr_description = processor.summarize_content(row['French scraped content'])
                    fr_keywords = processor.generate_keywords(row['French scraped content'])
                    # Perform SEO effectiveness evaluation for French content
                    status_text.text(f'Evaluating French SEO for URL {i+1}/{total_rows}')
                    fr_seo_evaluation = processor.review_metadata(row['French scraped content'], fr_description, fr_keywords)
                
                # Initialize translation variables
                translated_desc = ""
                translated_keywords = ""
                desc_review = ""
                keywords_review = ""
                
                # Only perform translation for english_french_pairs
                if input_type == "english_french_pairs":
                    # Translate English metadata to French
                    if en_description:
                        status_text.text(f'Translating English metadata for URL {i+1}/{total_rows}')
                        translated_desc = processor.translate_to_french(en_description, is_description=True)
                        translated_keywords = processor.translate_to_french(en_keywords, is_description=False)
                    
                    # Review French translations if we have French content
                    if row['French scraped content']:
                        status_text.text(f'Reviewing French translations for URL {i+1}/{total_rows}')
                        desc_review = processor.review_french_meta_description(
                            row['French scraped content'], translated_desc
                        )
                        keywords_review = processor.review_french_keywords(
                            row['French scraped content'], translated_keywords
                        )
                
                with st.expander(f"Metadata for URL {i+1}", expanded=False):
                    if en_description:
                        st.subheader("English Content")
                        st.text("Generated Description:")
                        st.code(en_description)
                        st.text("Generated Keywords:")
                        st.code(en_keywords)
                        
                        # Display SEO effectiveness evaluation for English content
                        st.text("SEO Effectiveness Evaluation:")
                        st.code(en_seo_evaluation)
                    
                    if fr_description:
                        st.subheader("French Content")
                        st.text("Generated Description:")
                        st.code(fr_description)
                        st.text("Generated Keywords:")
                        st.code(fr_keywords)
                        
                        # Display SEO effectiveness evaluation for French content
                        st.text("SEO Effectiveness Evaluation:")
                        st.code(fr_seo_evaluation)
                    
                    if input_type == "english_french_pairs":
                        st.subheader("Translated English to French")
                        st.text("Translated Description:")
                        st.code(translated_desc)
                        st.text("Translated Keywords:")
                        st.code(translated_keywords)
                        
                        st.subheader("Translation Reviews")
                        st.text("Meta Description Review:")
                        st.code(desc_review)
                        st.text("Keywords Review:")
                        st.code(keywords_review)
                
                # Different output format based on input type
                if input_type == "urls_only" or input_type == "urls_and_content":
                    # For English content
                    if row['English url'] and row['English scraped content']:
                        results.append({
                            'url': row['English url'],
                            'scraped_content': row['English scraped content'],
                            'generated_description': en_description,
                            'generated_keywords': en_keywords,
                            'model_recommendations': en_seo_evaluation
                        })
                    
                    # For French content
                    if row.get('French url') and row.get('French scraped content'):
                        results.append({
                            'url': row['French url'],
                            'scraped_content': row['French scraped content'],
                            'generated_description': fr_description,
                            'generated_keywords': fr_keywords,
                            'model_recommendations': fr_seo_evaluation
                        })
                else:
                    # For english_french_pairs
                    
                    results.append({
                        'English url': row['English url'],
                        'English scraped content': row['English scraped content'],
                        'French url': row['French url'],
                        'French scraped content': row['French scraped content'],
                        'English meta description': en_description,
                        'English meta keywords': en_keywords,
                        'English SEO evaluation': en_seo_evaluation,
                        'French meta description': fr_description,
                        'French keywords': fr_keywords,
                        'French SEO evaluation': fr_seo_evaluation,
                        'Translated English to french meta description': translated_desc,
                        'Translated English to french meta keywords': translated_keywords,
                        'French translation meta description review': desc_review,
                        'French translation meta keywords review': keywords_review
                    })
                
                progress_bar.progress((i + 1) / total_rows)
                
            except Exception as e:
                logger.error(f"Error processing row {i+1}: {str(e)}")
                st.error(f"Error processing URL {row.get('English url', '') or row.get('French url', '')}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in metadata generation: {str(e)}")
        st.error(f"An error occurred during metadata generation: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
        st.session_state.final_csv = None
    
    st.title('URL Content Scraper and Metadata Generator')
    
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY environment variable is not set")
        return
    
    # Only show input options if processing is not complete
    if not st.session_state.processing_complete:
        col1, col2 = st.columns(2)
        
        with col1:
            input_type = st.radio(
                "Select input type:",
                ["urls_only", "urls_and_content", "english_french_pairs"],
                format_func=lambda x: "URLs only (for scraping)" if x == "urls_only"
                         else "URLs and pre-scraped content" if x == "urls_and_content"
                         else "English/French URL pairs"
            )
        
        with col2:
            model_choice = st.selectbox(
                "Select AI model:",
                ["google/gemini-2.0-flash-001", "anthropic/claude-3.7-sonnet",
                 "mistralai/mixtral-8x7b-instruct"],
                format_func=lambda x: "Gemini Flash" if x == "google/gemini-2.0-flash-001"
                               else "Claude Sonnet 3.7" if x == "anthropic/claude-3.7-sonnet"
                               else "Mixtral 8x7B"
            )
            st.info(f"Using model: {model_choice}")
        
        if input_type == "urls_only":
            st.info("Upload a CSV file with URLs in the first column. Please include a header called 'urls'")
        elif input_type == "english_french_pairs":
            st.info("Upload a CSV file with two columns: 'english_url', 'french_url'. French URLs must contain '/fr/' in the path. English and French will be scraped, metadata generated in English, translated then the LLM will evaluate the French translation against the French scraped content.")
        else:
            st.info("Upload a CSV file with two columns: 'url' and 'scraped_content'")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.subheader('Processing Progress')
            scraping_progress = st.progress(0)
            scraping_status = st.empty()
            
            scraped_df = process_urls(uploaded_file, scraping_progress, scraping_status, input_type)
            
            if scraped_df is None:
                return
            
            st.success('URLs scraped successfully!')
            st.subheader('Generating Metadata')
            
            metadata_progress = st.progress(0)
            metadata_status = st.empty()
            model_status = st.empty()  
            
            final_df = generate_metadata(scraped_df, metadata_progress, metadata_status, model_status, input_type, model_choice)
            
            if final_df is not None:
                csv_data = final_df.to_csv(index=False)
                
                # Clear progress indicators
                metadata_progress.empty()
                metadata_status.empty()
                model_status.empty()
                
                # Show completion state
                st.success('Metadata generation complete!')
                st.download_button(
                    label="Download processed metadata CSV",
                    data=csv_data,
                    file_name="en-to-fr-metadata.csv" if input_type == "english_french_pairs" else "processed_metadata.csv",
                    mime="text/csv"
                )
                st.success("Processing completed! You may download your results.")
                
                # Update session state to prevent reprocessing
                st.session_state.processing_complete = True
                st.session_state.final_csv = csv_data
                st.session_state.input_type = input_type  # Store input type in session state
                st.session_state.model_choice = model_choice  # Store model choice in session state
            else:
                st.error("Failed to generate metadata. Please check the logs for details.")
    else:
        # Show saved completion state for subsequent visits
        st.success('Metadata generation complete!')
        st.download_button(
            label="Download processed metadata CSV",
            data=st.session_state.final_csv,
            file_name="en-to-fr-metadata.csv" if getattr(st.session_state, 'input_type', '') == "english_french_pairs" else "processed_metadata.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
