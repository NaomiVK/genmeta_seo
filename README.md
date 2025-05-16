# Metadata Generation System

## Project Overview

The Metadata Generation System is a powerful tool designed to generate high-quality metadata for web content in both English and French. The system scrapes web content, generates optimized meta descriptions and keywords, and evaluates their SEO effectiveness. It supports bilingual content through English/French URL pairs functionality, enabling seamless metadata generation for websites that maintain content in both languages.

Recent enhancements to the system include:

1. **English/French URL Pairs Functionality**: Process paired English and French versions of the same content, generate metadata for both languages, translate English metadata to French, and review the quality of French translations.

2. **SEO Effectiveness Evaluation**: Automatically evaluate the SEO effectiveness of generated metadata for all input types (urls_only, urls_and_content, english_french_pairs) in both English and French.

3. **Bug Fix for SEO Evaluation**: Fixed an issue with SEO evaluation variable initialization that ensures proper evaluation across all input types and prevents potential errors during processing.

## Features

### English/French URL Pairs Functionality

The English/French URL pairs functionality allows users to process paired English and French versions of the same content:

- **Paired Content Processing**: Scrape content from both English and French URLs
- **Validation**: Ensure French URLs contain '/fr/' in the path
- **Metadata Generation**: Generate metadata for both English and French content
- **Translation**: Translate English metadata to French
- **Translation Review**: Review French translations against French content for accuracy and cultural appropriateness

This feature is particularly valuable for websites and organizations that maintain content in both official languages of Canada, ensuring consistent and high-quality metadata across language versions.

### SEO Effectiveness Evaluation

The SEO effectiveness evaluation functionality provides comprehensive analysis of generated metadata:

- **Language Detection**: Automatically detect the language of content (English or French) through URL patterns and content analysis
- **Language-Specific Evaluation**: Apply appropriate SEO evaluation criteria based on detected language
- **Comprehensive Analysis**: Evaluate metadata for relevance to content, search engine optimization effectiveness, keyword density and placement, and user engagement potential
- **Actionable Recommendations**: Provide specific, actionable recommendations for improving metadata quality

This feature helps ensure that generated metadata is optimized for search engines and user engagement in both English and French.

### Language Detection

The system employs two methods for language detection:

1. **URL-based Detection**:
   - URLs containing '/fr/' in the path are identified as French content
   - All other URLs are assumed to be English content

2. **Content-based Detection**:
   - Uses the langdetect library to analyze the content text
   - Provides more accurate language determination when URL patterns are ambiguous
   - Ensures appropriate language-specific processing regardless of URL structure

## Usage Instructions

### Step-by-Step Guide

1. **Launch the application**:
   ```
   streamlit run main.py
   ```

2. **Select input type**:
   - Choose from "URLs only (for scraping)", "URLs and pre-scraped content", or "English/French URL pairs"

3. **Select AI model**:
   - Choose the model for metadata generation (default: Gemini Flash)
   - Note: Claude Sonnet is always used for SEO evaluation and French translation review regardless of this selection

4. **Upload your CSV file**:
   - Format depends on the selected input type (see below)

5. **Process the data**:
   - The system will automatically:
     - Scrape content if needed
     - Generate metadata
     - Evaluate SEO effectiveness
     - For English/French pairs: translate and review translations

6. **View and download results**:
   - Expand the metadata sections to view generated content and SEO evaluations
   - Click "Download processed metadata CSV" to save the results

### CSV Format Requirements

#### "urls_only" (with language detection and SEO evaluation)

- A CSV file with URLs in the first column
- Header should be "urls"
- Example:
  ```csv
  urls
  https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains.html
  https://www.canada.ca/fr/agence-revenu/services/impot/particuliers/sujets/tout-votre-declaration-revenus/declaration-revenus/remplir-declaration-revenus/revenu-personnel/ligne-12700-gains-capital.html
  ```

#### "urls_and_content" (with language detection and SEO evaluation)

- A CSV file with two columns: "url" and "scraped_content"
- Example:
  ```csv
  url,scraped_content
  https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains.html,"Capital gains are profits from the sale of a capital asset, such as shares, real estate, or valuable collectibles. In Canada, only 50% of capital gains are taxable."
  https://www.canada.ca/fr/agence-revenu/services/impot/particuliers/sujets/tout-votre-declaration-revenus/declaration-revenus/remplir-declaration-revenus/revenu-personnel/ligne-12700-gains-capital.html,"Les gains en capital sont des profits provenant de la vente d'une immobilisation, comme des actions, des biens immobiliers ou des objets de collection de valeur. Au Canada, seulement 50 % des gains en capital sont imposables."
  ```

#### "english_french_pairs" (with SEO evaluation)

- A CSV file with two columns: "english_url" and "french_url"
- French URLs must contain '/fr/' in the path
- Example:
  ```csv
  english_url,french_url
  https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains.html,https://www.canada.ca/fr/agence-revenu/services/impot/particuliers/sujets/tout-votre-declaration-revenus/declaration-revenus/remplir-declaration-revenus/revenu-personnel/ligne-12700-gains-capital.html
  https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/deductions-credits-expenses/line-21400-child-care-expenses.html,https://www.canada.ca/fr/agence-revenu/services/impot/particuliers/sujets/tout-votre-declaration-revenus/declaration-revenus/remplir-declaration-revenus/deductions-credits-depenses/ligne-21400-frais-garde-enfants.html
  ```

## Technical Details

### Metadata Generation Process

The metadata generation process follows these steps:

1. **Content Acquisition**:
   - For "urls_only": Scrape content from provided URLs
   - For "urls_and_content": Use pre-scraped content
   - For "english_french_pairs": Scrape content from both English and French URLs

2. **Language Detection**:
   - Determine language through URL patterns ('/fr/' indicates French)
   - Confirm language through content analysis using langdetect

3. **Metadata Generation**:
   - Generate meta descriptions using the `summarize_content()` method
   - Generate keywords using the `generate_keywords()` method
   - Both methods are language-aware and generate appropriate metadata for English or French content

4. **SEO Evaluation**:
   - The `review_metadata()` method evaluates the SEO effectiveness of the generated metadata
   - Different prompts are used for English and French content
   - The evaluation analyzes:
     - Relevance to content
     - Search engine optimization effectiveness
     - Keyword density and placement
     - User engagement potential

5. **Translation Process** (for English/French pairs):
   - English metadata is translated to French using the `translate_to_french()` method
   - Different prompts are used for descriptions and keywords:
     - Descriptions: Focus on natural language translation
     - Keywords: Focus on maintaining comma-separated format and industry terminology

6. **Translation Review** (for English/French pairs):
   - Compares translated metadata against the French content
   - Analyzes for translation accuracy, naturalness, SEO effectiveness, and cultural appropriateness
   - Provides specific suggestions for improvements

### Model Specifications

The system uses two different AI models:

1. **Metadata Generation and Translation**:
   - Default model: `google/gemini-2.0-flash-001` (Gemini Flash)
   - Can be changed by the user in the interface
   - Used for:
     - Generating English meta descriptions and keywords
     - Generating French meta descriptions and keywords
     - Translating English metadata to French

2. **SEO Effectiveness Evaluation and French Translation Review**:
   - Fixed model: `anthropic/claude-3.7-sonnet` (Claude Sonnet)
   - Cannot be changed by the user
   - Used for:
     - Evaluating SEO effectiveness of English metadata
     - Evaluating SEO effectiveness of French metadata
     - Reviewing French translation quality

## Testing

### Test Suites

The project includes comprehensive test suites to verify all functionality:

1. **English/French URL Pairs Functionality Tests**:
   - File: `test_en_fr_functionality.py`
   - Verifies CSV validation, URL processing, and metadata generation for English/French pairs

2. **SEO Effectiveness Evaluation Tests**:
   - File: `test_seo_evaluation.py`
   - Verifies language detection, SEO evaluation for different input types, and integration with existing functionality

### Test Data Files

The following test data files are provided:

1. `test_urls_only.csv`: URLs for scraping (mix of English and French URLs)
2. `test_urls_and_content.csv`: URLs and pre-scraped content (mix of English and French content)
3. `test_en_fr_pairs.csv`: Valid English/French URL pairs
4. `test_invalid_en_fr_pairs.csv`: Invalid CSV format (missing required columns)
5. `test_invalid_fr_url.csv`: Invalid French URL (missing '/fr/' in the path)

### Running the Tests

To run the tests:

1. Ensure the OPENROUTER_API_KEY environment variable is set:
   ```
   export OPENROUTER_API_KEY=your_api_key  # Linux/Mac
   set OPENROUTER_API_KEY=your_api_key     # Windows
   ```

2. Run all tests:
   ```
   python run_tests.py
   ```

3. Run specific test suites:
   ```
   python test_en_fr_functionality.py  # English/French pairs functionality tests
   python test_seo_evaluation.py       # SEO evaluation tests
   ```

4. Generate test data:
   ```
   python generate_mock_data.py        # Generate mock data for testing
   python generate_seo_test_data.py    # Generate SEO test data
   ```

### What the Tests Verify

The tests verify that:

1. **CSV Validation**:
   - Correctly identifies valid CSV formats for all input types
   - Rejects CSV files missing required columns

2. **URL Processing**:
   - Correctly processes URLs for all input types
   - Scrapes content from URLs when needed
   - Validates French URLs contain '/fr/'
   - Skips invalid pairs with appropriate warnings

3. **Language Detection**:
   - Correctly identifies French and English content based on URL patterns
   - Correctly identifies French and English content using content analysis

4. **Metadata Generation**:
   - Generates metadata for both English and French content
   - Translates English metadata to French (for English/French pairs)
   - Reviews French translations against French content (for English/French pairs)

5. **SEO Evaluation**:
   - Performs SEO effectiveness evaluation for all input types
   - Uses appropriate language-specific prompts for evaluation
   - Produces output with all required fields
   - Properly initializes evaluation variables to prevent errors

## Troubleshooting

If you encounter issues while using the Metadata Generation System, refer to the following common problems and solutions:

### SEO Evaluation Issues

- **Missing SEO Evaluation Results**: Ensure your content is substantial enough (at least 50 characters) and that you have proper network connectivity.
- **Variable Initialization Errors**: A recent fix addresses the issue where SEO evaluation variables weren't properly initialized. Update to the latest version if you encounter this problem.
- **Language Detection Problems**: For accurate language detection, ensure French URLs contain '/fr/' in the path and provide sufficient text content.

### API and Processing Issues

- **API Key Errors**: Verify that your OPENROUTER_API_KEY environment variable is correctly set.
- **Rate Limiting**: If you encounter API rate limit errors, try processing your data in smaller batches.
- **Processing Timeouts**: For large datasets, consider breaking the processing into multiple sessions.

For more detailed troubleshooting information, refer to the [SEO Effectiveness Evaluation Documentation](seo_effectiveness_evaluation_documentation.md).

## Conclusion

The Metadata Generation System provides a comprehensive solution for generating high-quality metadata for web content in both English and French. With its English/French URL pairs functionality and SEO effectiveness evaluation, it enables organizations to optimize their bilingual content for search engines and user engagement.

By following the instructions in this documentation, users can effectively utilize this system to improve the SEO and discoverability of their content in both English and French.