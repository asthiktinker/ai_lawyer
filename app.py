"""
Indian Legal Assistant - A comprehensive legal assistance tool with web scraping and AI analysis

This implementation combines:
1. Web scraping of Indian legal sources (Indian Kanoon, Legislative.gov.in, Supreme Court)

2. Legal text analysis using AI models
3. User-friendly Flask interface for accessing legal advice
"""

# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlparse
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
import logging
import redis
from functools import wraps
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Initialize Redis for caching (if available)
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis cache connected successfully")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, caching disabled")

# Initialize AI models
try:
    # Google Gemini setup
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Initialize with specific configuration for better legal responses
        gemini = genai.GenerativeModel(
            model_name='gemini-pro',
            generation_config={
                "temperature": 0.2,  # Lower temperature for more factual responses
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        AI_MODEL_AVAILABLE = True
        logger.info("Google Gemini AI model initialized with legal configuration")
    else:
        AI_MODEL_AVAILABLE = False
        logger.warning("Gemini API key not found, AI analysis will be limited")
        
    # Legal BERT classifier (optional)
    try:
        legal_classifier = pipeline(
            "text-classification", 
            model="nlpaueb/legal-bert-base-uncased",
            tokenizer="nlpaueb/legal-bert-base-uncased"
        )
        logger.info("Legal BERT classifier loaded")
        LEGAL_CLASSIFIER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Legal BERT classifier not loaded: {str(e)}")
        LEGAL_CLASSIFIER_AVAILABLE = False
        
except Exception as e:
    AI_MODEL_AVAILABLE = False
    LEGAL_CLASSIFIER_AVAILABLE = False
    logger.error(f"Error initializing AI models: {str(e)}")

# Legal websites configuration
LEGAL_SOURCES = {
    'indiankanoon': {
        'base_url': 'https://indiankanoon.org/search/',
        'params': {'formInput': '', 'pagenum': 1}
    },
    'legislative': {
        'base_url': 'https://legislative.gov.in/constitution-of-india/'
    },
    'supremecourt': {
        'base_url': 'https://main.sci.gov.in/judgments'
    }
}

# Legal Categories and Relevant Sections
LEGAL_CATEGORIES = {
    'RENT_DISPUTE': [
        "Transfer of Property Act Section 106",
        "Delhi Rent Control Act Section 6",
        "Maharashtra Rent Control Act Section 7"
    ],
    'PROPERTY_DISPUTE': [
        "Transfer of Property Act Section 54",
        "Registration Act Section 17",
        "Specific Relief Act Section 10"
    ],
    'CONTRACT_BREACH': [
        "Indian Contract Act Section 73",
        "Indian Contract Act Section 74",
        "Specific Relief Act Section 14"
    ],
    'FAMILY_DISPUTE': [
        "Hindu Marriage Act Section 13",
        "Special Marriage Act Section 27",
        "Hindu Succession Act Section 6"
    ],
    'CRIMINAL_OFFENSE': [
        "Indian Penal Code Section 319",
        "Code of Criminal Procedure Section 154",
        "Evidence Act Section 25"
    ]
}

# Helper Functions
def allowed_file(filename):
    """Check if uploaded file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rate_limited(max_per_minute):
    """Rate limiting decorator to prevent excessive scraping"""
    def decorator(func):
        last_called = {}
        min_interval = 60.0 / max_per_minute
        
        @wraps(func)
        def wrapper(url, *args, **kwargs):
            current_time = time.time()
            domain = urlparse(url).netloc
            
            if domain in last_called:
                elapsed = current_time - last_called[domain]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    logger.info(f"Rate limiting: Waiting {wait_time:.2f}s for {domain}")
                    time.sleep(wait_time)
            
            last_called[domain] = time.time()
            return func(url, *args, **kwargs)
        return wrapper
    return decorator

@rate_limited(10)  # Max 10 requests per minute per domain
def make_request(url, method='get', params=None, headers=None):
    """Make HTTP request with rate limiting and proper headers"""
    if headers is None:
        headers = {
            'User-Agent': 'LegalAssistant/1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.5'
        }
    
    try:
        if method.lower() == 'get':
            response = requests.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.post(url, data=params, headers=headers, timeout=30)
        
        # Log response code
        logger.info(f"Request to {url}: Status {response.status_code}")
        
        # Check if rate limited or blocked
        if response.status_code == 429:
            logger.warning(f"Rate limited by {url}")
            time.sleep(60)  # Wait longer if rate limited
            return None
        
        return response if response.status_code == 200 else None
    except Exception as e:
        logger.error(f"Error requesting {url}: {str(e)}")
        return None

def classify_legal_issue(text):
    """Classify the legal issue category"""
    if not text or len(text.strip()) < 10:
        return "UNKNOWN"
        
    # Use NLP model if available
    if LEGAL_CLASSIFIER_AVAILABLE:
        try:
            result = legal_classifier(text)[0]
            return result['label']
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
    
    # Fallback to keyword matching
    keywords = {
        'RENT_DISPUTE': ['rent', 'tenant', 'landlord', 'lease', 'eviction'],
        'PROPERTY_DISPUTE': ['property', 'sale deed', 'ownership', 'title', 'possession'],
        'CONTRACT_BREACH': ['contract', 'agreement', 'breach', 'terms', 'damages'],
        'FAMILY_DISPUTE': ['divorce', 'maintenance', 'custody', 'marriage', 'inheritance'],
        'CRIMINAL_OFFENSE': ['criminal', 'complaint', 'police', 'fir', 'offense', 'theft']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for category, category_keywords in keywords.items():
        score = sum(1 for keyword in category_keywords if keyword.lower() in text_lower)
        scores[category] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "UNKNOWN"

def generate_search_keywords(text):
    """Generate relevant search keywords from input text"""
    # Extract keywords based on legal domain
    keywords = []
    
    # Extract names of acts that might be mentioned
    act_pattern = r'([A-Z][a-z]+ (?:Act|Code|Rules|Bill)(?:\s+of\s+\d{4})?)'
    acts = re.findall(act_pattern, text)
    if acts:
        keywords.extend(acts)
    
    # Extract section numbers
    section_pattern = r'[Ss]ection\s+(\d+\w*)'
    sections = re.findall(section_pattern, text)
    if sections:
        keywords.extend([f"section {s}" for s in sections])
    
    # Add key phrases from issue classification
    issue_type = classify_legal_issue(text)
    if issue_type in LEGAL_CATEGORIES:
        # Add relevant legal sections as keywords
        relevant_sections = [s.split(' Section')[0] for s in LEGAL_CATEGORIES[issue_type][:2]]
        keywords.extend(relevant_sections)
    
    # Add additional context keywords
    context_keywords = []
    if 'rent' in text.lower() or 'tenant' in text.lower():
        context_keywords.append('rent control')
    if 'property' in text.lower() or 'sale' in text.lower():
        context_keywords.append('property transfer')
    if 'divorce' in text.lower() or 'marriage' in text.lower():
        context_keywords.append('marriage act')
    
    keywords.extend(context_keywords)
    
    # Ensure we have enough keywords by adding common words from input
    if len(keywords) < 3:
        common_legal_terms = ['legal', 'rights', 'court', 'law', 'judge', 'case']
        additional_words = [word for word in text.split() 
                          if len(word) > 3 and word.lower() not in common_legal_terms]
        keywords.extend(additional_words[:3])
    
    # Deduplicate and limit
    return list(dict.fromkeys(keywords))[:5]

def extract_text_from_file(file_path):
    """Extract text from uploaded files (PDF, TXT, images)"""
    try:
        file_ext = file_path.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_ext in ['png', 'jpg', 'jpeg']:
            text = pytesseract.image_to_string(Image.open(file_path))
        else:
            text = ""
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def scrape_indian_laws(query):
    """Scrape Indian legal websites for relevant information"""
    legal_data = []
    
    # Check cache first
    cache_key = f"legal_search:{hash(query)}"
    if REDIS_AVAILABLE:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return json.loads(cached_data)
    
    # Generate search keywords
    keywords = generate_search_keywords(query)
    search_term = " ".join(keywords)
    logger.info(f"Generated search keywords: {keywords}")
    
    try:
        # 1. Scrape Indian Kanoon
        ik_params = LEGAL_SOURCES['indiankanoon']['params'].copy()
        ik_params['formInput'] = search_term
        
        response = make_request(
            LEGAL_SOURCES['indiankanoon']['base_url'],
            params=ik_params
        )
        
        if response:
            soup = BeautifulSoup(response.text, 'lxml')
            results = soup.select('.result')[:5]  # Get top 5 results
            
            for result in results:
                try:
                    title_elem = result.select_one('.result_title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = "https://indiankanoon.org" + title_elem['href'] if title_elem['href'].startswith('/doc') else title_elem['href']
                    
                    citation_elem = result.select_one('.docsource')
                    citation = citation_elem.text.strip() if citation_elem else "Citation not available"
                    
                    sections = [s.text.strip() for s in result.select('.doc_cite')]
                    
                    # Get a short snippet
                    snippet_elem = result.select_one('.snippet')
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    
                    case = {
                        'source': 'Indian Kanoon',
                        'title': title,
                        'citation': citation,
                        'sections': sections,
                        'snippet': snippet[:200] + '...' if len(snippet) > 200 else snippet,
                        'link': link
                    }
                    legal_data.append(case)
                except Exception as e:
                    logger.error(f"Error parsing IndianKanoon result: {str(e)}")
                    continue

        # 2. Scrape Legislative.gov.in for relevant acts
        issue_type = classify_legal_issue(query)
        if issue_type in LEGAL_CATEGORIES:
            act_names = set()
            for section in LEGAL_CATEGORIES[issue_type]:
                act_name = section.split(' Section')[0].strip()
                act_names.add(act_name)
            
            for act_name in list(act_names)[:2]:  # Top 2 relevant acts
                act_search_url = f"https://legislative.gov.in/search/{act_name.replace(' ', '+')}"
                response = make_request(act_search_url)
                
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = soup.select('.view-content .views-row')[:2]
                    
                    for result in results:
                        try:
                            title_elem = result.select_one('h2 a')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link = title_elem['href']
                            if not link.startswith('http'):
                                link = "https://legislative.gov.in" + link
                            
                            legal_data.append({
                                'source': 'Legislative.gov.in',
                                'title': title,
                                'type': 'statute',
                                'link': link
                            })
                        except Exception as e:
                            logger.error(f"Error parsing legislative result: {str(e)}")
                            continue

        # 3. Scrape Supreme Court website (if relevant)
        if issue_type in ['PROPERTY_DISPUTE', 'CRIMINAL_OFFENSE', 'CONTRACT_BREACH']:
            # This is more complex as SC site might use forms/JS
            # For this implementation, we'll include a placeholder
            legal_data.append({
                'source': 'Supreme Court of India',
                'title': f"Latest judgments related to {issue_type.replace('_', ' ').title()}",
                'type': 'judgment',
                'link': 'https://main.sci.gov.in/judgments'
            })
    
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
    
    # Cache results
    if REDIS_AVAILABLE and legal_data:
        redis_client.setex(
            cache_key,
            3600,  # Cache for 1 hour
            json.dumps(legal_data)
        )
    
    return legal_data

def generate_legal_advice(query, legal_data):
    """Generate legal advice using AI"""
    try:
        # Check if AI is available
        if not AI_MODEL_AVAILABLE:
            logger.warning("AI model not available, using fallback analysis")
            return generate_fallback_analysis(query)
            
        # Get issue type and applicable laws
        issue_type = classify_legal_issue(query)
        applicable_laws = LEGAL_CATEGORIES.get(issue_type, [])
        
        # Prepare context from scraped legal data
        legal_context = ""
        for i, item in enumerate(legal_data[:3]):  # Use top 3 results for context
            if 'title' in item:
                legal_context += f"Legal Source {i+1}: {item['title']}\n"
            if 'sections' in item and item['sections']:
                legal_context += f"Sections: {', '.join(item['sections'])}\n"
            if 'snippet' in item and item['snippet']:
                legal_context += f"Details: {item['snippet']}\n\n"
        
        # Generate prompt for Gemini
        prompt = f"""
        As a legal assistant specializing in Indian law, provide a concise analysis of the following legal issue:
        
        User Query: {query}
        
        Issue Classification: {issue_type.replace('_', ' ').title()}
        
        Applicable Laws:
        {' '.join(applicable_laws)}
        
        Legal Context from Indian Legal Sources:
        {legal_context}
        
        Provide a 3-part response:
        1. A brief summary of the legal issue (2-3 sentences)
        2. The key applicable laws and their relevance (2-3 sentences)
        3. Initial steps the person should consider (3-4 bullet points)
        
        Keep your advice practical, accurate, and ethical. Include a disclaimer that this is general information and not specific legal advice.
        """
        
        try:
            # Call Gemini API with proper error handling and timeout
            response = gemini.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Log successful API call
            logger.info("Successfully received response from Gemini API")
            
            # Check if we have a valid response
            if not hasattr(response, 'text') or not response.text:
                logger.error("Empty response from Gemini API")
                return generate_fallback_analysis(query)
                
            response_text = response.text
            
            # Parse the response with more robust handling
            sections = response_text.split('\n\n')
            
            summary = ""
            laws_text = ""
            recommendations = []
            
            # More robust parsing
            summary_markers = ["summary", "1.", "issue:"]
            laws_markers = ["applicable laws", "key laws", "2.", "laws:"]
            recommendation_markers = ["steps", "consider", "3.", "recommend", "action"]
            
            for section in sections:
                section_lower = section.lower()
                
                # Match summary section
                if any(marker in section_lower for marker in summary_markers):
                    # Extract content after colon or number
                    if ":" in section:
                        summary = section.split(":", 1)[1].strip()
                    else:
                        summary = re.sub(r"^1\.\s*", "", section).strip()
                
                # Match applicable laws section
                elif any(marker in section_lower for marker in laws_markers):
                    if ":" in section:
                        laws_text = section.split(":", 1)[1].strip()
                    else:
                        laws_text = re.sub(r"^2\.\s*", "", section).strip()
                
                # Match recommendations section and extract bullet points
                elif any(marker in section_lower for marker in recommendation_markers):
                    # Handle bullet points or numbered lists
                    lines = section.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith("-") or line.startswith("•") or 
                                     line.startswith("*") or re.match(r"^\d+\.", line)):
                            # Clean up the bullet point
                            clean_line = re.sub(r"^[-•*\d\.]+\s*", "", line).strip()
                            if clean_line:
                                recommendations.append(clean_line)
                    
                    # If no bullet points found but section matches, use whole text
                    if not recommendations and len(lines) > 1:
                        clean_text = re.sub(r"^3\.\s*", "", section).strip()
                        recommendations = [line.strip() for line in clean_text.split("\n") 
                                          if line.strip() and not any(m in line.lower() for m in recommendation_markers)]
            
            # Ensure we have all required parts or use fallbacks
            if not summary:
                summary = f"This appears to be a {issue_type.replace('_', ' ').lower()} matter that may involve multiple provisions of Indian law."
                
            if not laws_text and applicable_laws:
                laws_text = f"The key laws applicable to this situation include {', '.join(applicable_laws[:3])}. These laws establish the rights and responsibilities in this type of legal matter."
                
            # Ensure we have recommendations
            if not recommendations:
                recommendations = [
                    "Consult with a qualified legal professional about your specific situation",
                    "Gather all relevant documentation including agreements, communications, and evidence",
                    "Learn about your rights under the applicable legal provisions",
                    "Consider alternative dispute resolution methods before litigation"
                ]
            
            return {
                'summary': summary,
                'issue_type': issue_type.replace('_', ' ').title(),
                'applicable_laws': applicable_laws,
                'laws_analysis': laws_text,
                'recommendations': recommendations[:4],  # Limit to top 4
                'disclaimer': "This is general information only, not specific legal advice. Please consult with a qualified legal professional."
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            return generate_fallback_analysis(query)
            
    except Exception as e:
        logger.error(f"Error generating legal advice: {str(e)}")
        return generate_fallback_analysis(query)

def generate_fallback_analysis(query):
    """Generate fallback analysis when AI is unavailable or fails"""
    issue_type = classify_legal_issue(query)
    applicable_laws = LEGAL_CATEGORIES.get(issue_type, [])
    
    # Create meaningful fallback content based on classification
    summaries = {
        'RENT_DISPUTE': "This appears to be a rental dispute involving tenant and landlord rights. Indian rent control acts specify terms for rent increases, eviction procedures, and maintenance responsibilities.",
        'PROPERTY_DISPUTE': "This involves a property ownership or transfer dispute. Indian property law covers registration requirements, title verification, and legal remedies for property claims.",
        'CONTRACT_BREACH': "This concerns a potential breach of contract situation. Indian contract law provides remedies including damages, specific performance, and contractual obligations.",
        'FAMILY_DISPUTE': "This relates to a family law matter, potentially involving marriage, divorce, or inheritance. Personal laws in India vary based on religion and govern family relationships.",
        'CRIMINAL_OFFENSE': "This appears to involve potential criminal charges or proceedings. Indian criminal procedure requires following specific steps for complaints, investigation, and trial.",
        'UNKNOWN': "This legal matter may involve multiple areas of Indian law. A detailed analysis would require consultation with a specialized legal professional."
    }
    
    recommendations = {
        'RENT_DISPUTE': [
            "Review your rental agreement for specific terms about rent increases, maintenance, and termination",
            "Document all communications with your landlord in writing",
            "Check applicable rent control laws in your specific city/state",
            "Consider mediation before pursuing legal action"
        ],
        'PROPERTY_DISPUTE': [
            "Gather all property documents including sale deed, registration papers, and tax receipts",
            "Verify the property's legal history and encumbrances through proper channels",
            "Consider sending a legal notice before filing a case",
            "Consult with a property law specialist about boundary or ownership disputes"
        ],
        'CONTRACT_BREACH': [
            "Document all evidence of the breach of contract",
            "Calculate potential damages with supporting documentation",
            "Send a formal notice to the breaching party",
            "Explore arbitration options as specified in your contract"
        ],
        'FAMILY_DISPUTE': [
            "Gather all relevant certificates and financial documents",
            "Consider family counseling as a first step",
            "Understand your rights under personal laws applicable to your religion",
            "Explore mediation for amicable settlement of disputes"
        ],
        'CRIMINAL_OFFENSE': [
            "File a police complaint (FIR) at the appropriate police station",
            "Maintain copies of all documents related to the case",
            "Seek immediate legal representation",
            "Do not make statements without legal counsel present"
        ],
        'UNKNOWN': [
            "Document all facts and evidence related to your legal matter",
            "Consult with a legal professional specializing in the relevant field",
            "Understand the timeline and potential costs of legal proceedings",
            "Consider alternative dispute resolution methods if appropriate"
        ]
    }
    
    laws_analysis = {
        'RENT_DISPUTE': "Rent control acts vary by state and provide protections against arbitrary eviction and excessive rent increases. The Transfer of Property Act also governs landlord-tenant relationships.",
        'PROPERTY_DISPUTE': "Property disputes are governed by the Transfer of Property Act, Registration Act and specific local laws. These establish requirements for valid transfers and remedies for disputes.",
        'CONTRACT_BREACH': "The Indian Contract Act specifies remedies for breach including damages to compensate the injured party. Specific Relief Act provides for enforcement of contracts in certain cases.",
        'FAMILY_DISPUTE': "Family matters are governed by personal laws specific to religion and the Special Marriage Act. These laws establish rights and procedures for marriage, divorce, and inheritance.",
        'CRIMINAL_OFFENSE': "Criminal matters follow procedures in the Criminal Procedure Code and substantive law in the Indian Penal Code. These establish offenses and the process for investigation and trial.",
        'UNKNOWN': "Multiple laws may apply based on the specific details of your situation. Indian law provides various remedies and procedures depending on the nature of the legal issue."
    }
    
    return {
        'summary': summaries.get(issue_type, summaries['UNKNOWN']),
        'issue_type': issue_type.replace('_', ' ').title(),
        'applicable_laws': applicable_laws,
        'laws_analysis': laws_analysis.get(issue_type, laws_analysis['UNKNOWN']),
        'recommendations': recommendations.get(issue_type, recommendations['UNKNOWN']),
        'disclaimer': "This is general information only, not specific legal advice. Please consult with a qualified legal professional."
    }
#flakes routes
@app.route('/')
def home():
    # Only render template without passing analysis
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process user query and provide legal analysis"""
    try:
        # Get input data
        text_input = request.form.get('text', '')
        uploaded_file = request.files.get('file')
        file_text = ''

        # Process uploaded file if any
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save and process file
            uploaded_file.save(filepath)
            file_text = extract_text_from_file(filepath)

        # Combine inputs
        full_text = f"{text_input}\n\n{file_text}".strip()
        
        if not full_text:
            return render_template('result.html', 
                error="Please provide text input or upload a document")

        # Log the query (without PII)
        query_len = len(full_text)
        logger.info(f"Processing query of length {query_len} characters")
        
        # Scrape legal information
        legal_data = scrape_indian_laws(full_text)
        
        # Generate legal analysis
        analysis = generate_legal_advice(full_text, legal_data)
        analysis['cases'] = legal_data

        return render_template('result.html', analysis=analysis)

    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        return render_template('result.html', error=f"An error occurred: {str(e)}")

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for legal analysis"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
            
        query = data['query']
        legal_data = scrape_indian_laws(query)
        analysis = generate_legal_advice(query, legal_data)
        
        return jsonify({
            'analysis': analysis,
            'legal_sources': legal_data
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Check if Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR available")
    except:
        logger.warning("Tesseract OCR not found. Image processing will be limited.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False') == 'True')
