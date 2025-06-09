from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import uuid
import logging
import os
import json
from typing import TypedDict, List, Dict, Optional, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import re
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for cross-origin requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup OpenAI and Gmail SMTP credentials from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
if not GMAIL_USER or not GMAIL_PASSWORD:
    raise ValueError("GMAIL_USER or GMAIL_PASSWORD environment variable not set")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    max_tokens=2000,
    api_key=OPENAI_API_KEY
)

# State definition
class State(TypedDict):
    job_description: str
    resume: str
    candidate_id: str
    position_title: str
    experience_level: str
    skill_match: str
    skill_score: float
    experience_score: float
    cultural_fit_score: float
    technical_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    salary_expectation: Optional[str]
    availability: Optional[str]
    response: str
    detailed_feedback: str
    confidence_score: float
    processing_time: float
    screening_metadata: Dict
    candidate_name: Optional[str]
    candidate_email: Optional[str]

def extract_pdf_text(file_stream) -> Optional[str]:
    """Extract text from a PDF file stream and clean it for better regex matching"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            # Clean up the text: remove non-printable characters and normalize whitespace
            cleaned_text = re.sub(r'[^\x20-\x7E\n]', ' ', page_text)  # Replace non-ASCII with space
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
            text += cleaned_text
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return None

def extract_candidate_info(resume_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract candidate name and email from resume text using improved regex"""
    # Name pattern: Match names at the start of the text, allowing for spaces and multiple words
    name_pattern = r'^[A-Z][a-zA-Z\s]*(?:[A-Z][a-zA-Z]+)?'
    
    # Clean the resume text: normalize spaces around @ symbol for email matching
    cleaned_text = re.sub(r'\s*@\s*', '@', resume_text.strip())  # Remove spaces around @
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize other whitespace
    
    # Email pattern: Match email addresses, allowing for potential spaces around @
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    
    name_match = re.search(name_pattern, cleaned_text, re.MULTILINE)
    email_match = re.search(email_pattern, cleaned_text)
    
    candidate_name = name_match.group(0).strip() if name_match else None
    candidate_email = email_match.group(0).strip() if email_match else None
    
    return candidate_name, candidate_email

def send_result_email(candidate_name: Optional[str], candidate_email: Optional[str], 
                     position_title: str, screening_result: Dict):
    """Send screening results via email using Gmail SMTP"""
    if not candidate_email:
        logger.warning("No email found for candidate, skipping email notification")
        return {'status': 'error', 'message': 'No candidate email provided'}
    
    try:
        # Setup email
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = candidate_email
        msg['Subject'] = f'Screening Results for {position_title} Position'
        
        # Determine selection status based on response
        result_message = screening_result['response']
        is_selected = result_message.startswith("SHORTLISTED") or result_message.startswith("QUALIFIED")
        status = "selected for an interview" if is_selected else "not selected at this time"
        
        # Email body
        greeting = f"Dear {candidate_name or 'Candidate'}," 
        feedback = screening_result['detailed_feedback']
        
        body = f"""
{greeting}

Thank you for applying for the {position_title} position. We have completed the initial screening process, and we would like to share the results with you.

Screening Result: You have been {status}.
{feedback}
{f"Next Steps: Our team will reach out to schedule your {'technical' if 'technical_interview' in screening_result['screening_metadata']['decision'] else 'HR'} interview." if is_selected else "We appreciate your interest and encourage you to apply for other suitable positions in the future."}
Best regards,
Hiring Team
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {candidate_email}")
        return {'status': 'success', 'message': f'Email sent to {candidate_email}'}
    
    except Exception as e:
        logger.error(f"Error sending email to {candidate_email}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

# Processing functions
def extract_skills_and_score(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Analyze the resume against job requirements and provide detailed skill assessment.
        
        Job Requirements: {job_description}
        Candidate Resume: {resume}
        
        Provide a JSON response with:
        {{
            "technical_skills": ["skill1", "skill2", "skill3"],
            "missing_skills": ["missing1", "missing2"],
            "skill_score": 85.5,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "confidence_score": 92.0
        }}
        
        Score from 0-100 based on skill alignment, experience relevance, and requirements match."""
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "job_description": state["job_description"],
        "resume": state["resume"]
    }).content
    
    try:
        json_match = re.search(r'\{[\s\S]*?\}', result, re.MULTILINE)
        if not json_match:
            raise ValueError("No valid JSON object found in response")
        
        json_str = json_match.group(0)
        skill_data = json.loads(json_str.strip())
        
        # Extract candidate info
        candidate_name, candidate_email = extract_candidate_info(state["resume"])
        
        return {  
            "technical_skills": skill_data.get("technical_skills", []),
            "missing_skills": skill_data.get("missing_skills", []),
            "skill_score": float(skill_data.get("skill_score", 0)),
            "strengths": skill_data.get("strengths", []),
            "weaknesses": skill_data.get("weaknesses", []),
            "confidence_score": float(skill_data.get("confidence_score", 0)),
            "candidate_name": candidate_name,
            "candidate_email": candidate_email
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing skill assessment: {e}. Raw response: {result}")
        return {
            "technical_skills": [],
            "missing_skills": [],
            "skill_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "confidence_score": 0.0,
            "candidate_name": None,
            "candidate_email": None
        }

def categorize_experience_advanced(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Analyze candidate experience with detailed scoring.
        
        Job Requirements: {job_description}
        Candidate Resume: {resume}
        
        Provide JSON response:
        {{
            "experience_level": "Entry-level",
            "experience_score": 75.5,
            "years_experience": 5,
            "relevant_experience": 4,
            "leadership_experience": true,
            "domain_expertise": "High"
        }}
        
        Experience levels: Entry-level (0-2 years), Mid-level (3-5 years), Senior-level (6-10 years), Expert-level (10+ years)"""
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "job_description": state["job_description"],
        "resume": state["resume"]
    }).content
    
    try:
        json_match = re.search(r'\{[\s\S]*?\}', result)
        if not json_match:
            raise ValueError("No valid JSON object found in response")
        
        json_str = json_match.group(0)
        exp_data = json.loads(json_str.strip())
        return {
            "experience_level": exp_data.get("experience_level", "Unknown"),
            "experience_score": float(exp_data.get("experience_score", 0))
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing experience assessment: {e}. Raw response: {result}")
        return {"experience_level": "Unknown", "experience_score": 0.0}

def assess_cultural_fit(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Assess cultural fit based on resume indicators.
        
        Job Description: {job_description}
        Resume: {resume}
        
        Analyze these aspects and provide a score 0-100:
        - Communication style indicators
        - Team collaboration mentions  
        - Learning agility signals
        - Problem-solving approach
        - Innovation/creativity indicators
        - Leadership potential
        
        Provide brief reasoning and a cultural_fit_score between 0-100."""
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "job_description": state["job_description"],
        "resume": state["resume"]
    }).content
    
    cultural_score = 70.0
    try:
        scores = re.findall(r'\b(\d{1,2}(?:\.\d)?|\d{3})\b', result)
        if scores:
            potential_scores = [float(s) for s in scores if 0 <= float(s) <= 100]
            if potential_scores:
                cultural_score = potential_scores[0]
    except:
        pass
    
    return {"cultural_fit_score": cultural_score}

def determine_skill_match_advanced(state: State) -> State:
    skill_weight = 0.4
    experience_weight = 0.35
    cultural_weight = 0.25
    
    skill_score = state.get("skill_score", 0)
    experience_score = state.get("experience_score", 0)
    cultural_score = state.get("cultural_fit_score", 0)
    
    composite_score = (
        skill_score * skill_weight +
        experience_score * experience_weight +
        cultural_score * cultural_weight
    )
    
    skill_match = "Match" if (
        composite_score >= 70 and
        skill_score >= 60 and
        experience_score >= 50
    ) else "No Match"
    
    return {"skill_match": skill_match}

def generate_detailed_feedback(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Generate comprehensive candidate feedback report.
        
        Candidate Assessment:
        - Technical Skills: {technical_skills}
        - Missing Skills: {missing_skills}
        - Strengths: {strengths}
        - Weaknesses: {weaknesses}
        - Skill Score: {skill_score}/100
        - Experience Score: {experience_score}/100
        - Cultural Fit: {cultural_fit_score}/100
        
        Create a professional, constructive feedback report (200-300 words)."""
    )
    
    chain = prompt | llm
    detailed_feedback = chain.invoke({
        "technical_skills": state.get("technical_skills", []),
        "missing_skills": state.get("missing_skills", []),
        "strengths": state.get("strengths", []),
        "weaknesses": state.get("weaknesses", []),
        "skill_score": state.get("skill_score", 0),
        "experience_score": state.get("experience_score", 0),
        "cultural_fit_score": state.get("cultural_fit_score", 0)
    }).content
    
    return {"detailed_feedback": detailed_feedback}

def schedule_technical_interview(state: State) -> State:
    skill_score = state.get("skill_score", 0)
    return {
        "response": f"SHORTLISTED: High-quality candidate with {skill_score:.1f}% skill match. Recommended for technical interview.",
        "screening_metadata": {
            "decision": "technical_interview",
            "priority": "high" if skill_score > 85 else "standard",
            "interview_type": "technical_assessment",
            "estimated_duration": "90 minutes"
        }
    }

def schedule_hr_interview_only(state: State) -> State:
    skill_score = state.get("skill_score", 0)
    return {
        "response": f"QUALIFIED: Good candidate match ({skill_score:.1f}% skills). Scheduling HR interview first.",
        "screening_metadata": {
            "decision": "hr_interview",
            "priority": "standard",
            "interview_type": "behavioral_assessment",
            "notes": "Requires skill validation"
        }
    }

def escalate_to_hiring_manager(state: State) -> State:
    return {
        "response": f"ESCALATED: Senior candidate with mixed alignment. Requires hiring manager review.",
        "screening_metadata": {
            "decision": "escalation",
            "reason": "senior_experience_skill_gap",
            "requires_manual_review": True,
            "escalation_priority": "medium"
        }
    }

def reject_with_feedback(state: State) -> State:
    skill_score = state.get("skill_score", 0)
    return {
        "response": f"NOT QUALIFIED: Candidate doesn't meet minimum requirements ({skill_score:.1f}% match).",
        "screening_metadata": {
            "decision": "rejected",
            "reason": "insufficient_qualification",
            "feedback_provided": True,
            "reapplication_eligible": skill_score > 40
        }
    }

def intelligent_routing(state: State) -> str:
    skill_score = state.get("skill_score", 0)
    experience_score = state.get("experience_score", 0)
    experience_level = state.get("experience_level", "")
    
    if skill_score >= 80 and experience_score >= 70:
        return "schedule_technical_interview"
    elif skill_score >= 65 and experience_score >= 60:
        return "schedule_hr_interview_only"
    elif experience_level == "Senior-level" and skill_score >= 50:
        return "escalate_to_hiring_manager"
    else:
        return "reject_with_feedback"

def setup_advanced_workflow():
    workflow = StateGraph(State)
    
    workflow.add_node("extract_skills", extract_skills_and_score)
    workflow.add_node("categorize_experience", categorize_experience_advanced)
    workflow.add_node("assess_cultural_fit", assess_cultural_fit)
    workflow.add_node("determine_match", determine_skill_match_advanced)
    workflow.add_node("generate_feedback", generate_detailed_feedback)
    workflow.add_node("schedule_technical_interview", schedule_technical_interview)
    workflow.add_node("schedule_hr_interview_only", schedule_hr_interview_only)
    workflow.add_node("escalate_to_hiring_manager", escalate_to_hiring_manager)
    workflow.add_node("reject_with_feedback", reject_with_feedback)
    
    workflow.add_edge(START, "extract_skills")
    workflow.add_edge("extract_skills", "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_cultural_fit")
    workflow.add_edge("assess_cultural_fit", "determine_match")
    workflow.add_edge("determine_match", "generate_feedback")
    workflow.add_conditional_edges("generate_feedback", intelligent_routing)
    workflow.add_edge("schedule_technical_interview", END)
    workflow.add_edge("schedule_hr_interview_only", END)
    workflow.add_edge("escalate_to_hiring_manager", END)
    workflow.add_edge("reject_with_feedback", END)
    
    return workflow.compile()

# Serve the frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# API Endpoints
@app.route('/api/screening/single', methods=['POST'])
def single_screening():
    """API endpoint for screening a single candidate with text resume"""
    try:
        data = request.get_json()
        if not data or 'job_description' not in data or 'resume' not in data:
            return jsonify({
                'error': 'Missing required fields: job_description and resume'
            }), 400

        start_time = datetime.now()
        candidate_id = data.get('candidate_id', f"CAND_{uuid.uuid4().hex[:8]}")
        position_title = data.get('position_title', 'Software Developer')
        
        app = setup_advanced_workflow()
        results = app.invoke({
            "job_description": data['job_description'],
            "resume": data['resume'],
            "candidate_id": candidate_id,
            "position_title": position_title,
            "processing_time": 0,
            "screening_metadata": {}
        })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "candidate_id": candidate_id,
            "position_title": position_title,
            "experience_level": results.get("experience_level", "Unknown"),
            "skill_match": results.get("skill_match", "Unknown"),
            "skill_score": results.get("skill_score", 0),
            "experience_score": results.get("experience_score", 0),
            "cultural_fit_score": results.get("cultural_fit_score", 0),
            "technical_skills": results.get("technical_skills", []),
            "missing_skills": results.get("missing_skills", []),
            "strengths": results.get("strengths", []),
            "weaknesses": results.get("weaknesses", []),
            "response": results.get("response", ""),
            "detailed_feedback": results.get("detailed_feedback", ""),
            "confidence_score": results.get("confidence_score", 0),
            "processing_time": processing_time,
            "screening_metadata": results.get("screening_metadata", {}),
            "candidate_name": results.get("candidate_name", None),
            "candidate_email": results.get("candidate_email", None),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Completed screening for candidate {candidate_id}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in single screening: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/screening/pdf', methods=['POST'])
def pdf_screening():
    """API endpoint for screening a candidate with PDF resume"""
    try:
        if 'resume' not in request.files or 'job_description' not in request.form:
            return jsonify({
                'error': 'Missing required fields: job_description and resume PDF file'
            }), 400

        pdf_file = request.files['resume']
        job_description = request.form['job_description']
        candidate_id = request.form.get('candidate_id', f"CAND_{uuid.uuid4().hex[:8]}")
        position_title = request.form.get('position_title', 'Software Developer')

        # Extract text from PDF
        resume_text = extract_pdf_text(pdf_file)
        if not resume_text:
            return jsonify({
                'error': 'Failed to extract text from PDF'
            }), 400

        start_time = datetime.now()
        app = setup_advanced_workflow()
        results = app.invoke({
            "job_description": job_description,
            "resume": resume_text,
            "candidate_id": candidate_id,
            "position_title": position_title,
            "processing_time": 0,
            "screening_metadata": {}
        })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "candidate_id": candidate_id,
            "position_title": position_title,
            "experience_level": results.get("experience_level", "Unknown"),
            "skill_match": results.get("skill_match", "Unknown"),
            "skill_score": results.get("skill_score", 0),
            "experience_score": results.get("experience_score", 0),
            "cultural_fit_score": results.get("cultural_fit_score", 0),
            "technical_skills": results.get("technical_skills", []),
            "missing_skills": results.get("missing_skills", []),
            "strengths": results.get("strengths", []),
            "weaknesses": results.get("weaknesses", []),
            "response": results.get("response", ""),
            "detailed_feedback": results.get("detailed_feedback", ""),
            "confidence_score": results.get("confidence_score", 0),
            "processing_time": processing_time,
            "screening_metadata": results.get("screening_metadata", {}),
            "candidate_name": results.get("candidate_name", None),
            "candidate_email": results.get("candidate_email", None),
            "timestamp": datetime.now().isoformat(),
            "extracted_resume_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        }
        
        logger.info(f"Completed PDF screening for candidate {candidate_id}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in PDF screening: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/screening/batch', methods=['POST'])
def batch_screening_api():
    """API endpoint for batch screening multiple candidates"""
    try:
        data = request.get_json()
        if not data or 'job_description' not in data or 'candidates' not in data:
            return jsonify({
                'error': 'Missing required fields: job_description and candidates'
            }), 400

        results = []
        for candidate in data['candidates']:
            start_time = datetime.now()
            candidate_id = candidate.get('id', f"CAND_{uuid.uuid4().hex[:8]}")
            position_title = candidate.get('position', 'Software Developer')
            
            app = setup_advanced_workflow()
            result = app.invoke({
                "job_description": data['job_description'],
                "resume": candidate['resume'],
                "candidate_id": candidate_id,
                "position_title": position_title,
                "processing_time": 0,
                "screening_metadata": {}
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "candidate_id": candidate_id,
                "position_title": position_title,
                "experience_level": result.get("experience_level", "Unknown"),
                "skill_match": result.get("skill_match", "Unknown"),
                "skill_score": result.get("skill_score", 0),
                "experience_score": result.get("experience_score", 0),
                "cultural_fit_score": result.get("cultural_fit_score", 0),
                "technical_skills": result.get("technical_skills", []),
                "missing_skills": result.get("missing_skills", []),
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", []),
                "response": result.get("response", ""),
                "detailed_feedback": result.get("detailed_feedback", ""),
                "confidence_score": result.get("confidence_score", 0),
                "processing_time": processing_time,
                "screening_metadata": result.get("screening_metadata", {}),
                "candidate_name": result.get("candidate_name", None),
                "candidate_email": result.get("candidate_email", None),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(response)
        
        total_matches = sum(1 for r in results if r['skill_match'] == 'Match')
        avg_skill_score = sum(r['skill_score'] for r in results) / len(results) if results else 0
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
        
        response = {
            "results": results,
            "summary": {
                "total_candidates": len(results),
                "matches_found": total_matches,
                "match_rate": (total_matches/len(results)*100) if results else 0,
                "avg_skill_score": avg_skill_score,
                "avg_processing_time": avg_processing_time
            }
        }
        
        logger.info(f"Completed batch screening for {len(results)} candidates")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch screening: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-confirmation', methods=['POST'])
def send_confirmation_email():
    """API endpoint to send confirmation email for a candidate"""
    try:
        data = request.get_json()
        if not data or 'candidate_id' not in data or 'position_title' not in data or 'response' not in data:
            return jsonify({
                'error': 'Missing required fields: candidate_id, position_title, response'
            }), 400

        candidate_id = data['candidate_id']
        position_title = data['position_title']
        screening_result = data['response']
        candidate_name = data.get('candidate_name')
        candidate_email = data.get('candidate_email')

        # Send email
        email_result = send_result_email(
            candidate_name,
            candidate_email,
            position_title,
            screening_result
        )
        
        return jsonify(email_result), 200
        
    except Exception as e:
        logger.error(f"Error sending confirmation email: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)