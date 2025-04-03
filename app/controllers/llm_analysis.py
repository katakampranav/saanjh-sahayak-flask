import google.generativeai as genai
from config import Config

# Configure Gemini API key
genai.configure(api_key=Config.GEMINI_API_KEY)
    
def analyze_medical_report(report):
    """Generates an upcycling idea using Gemini AI."""
    prompt = f'''
        You are an advanced AI medical report analyzer specializing in elderly care. Based on the given **medical report content**, generate a structured analysis with the following **clear and precise outputs**:

        ---

        ## **1️⃣ DETAILED ANALYSIS**
        - Provide a **comprehensive** breakdown of the patient's condition.
        - Explain **key findings, test results, and their medical significance**.
        - Highlight any **critical observations** requiring immediate attention.

        ## **2️⃣ PRECAUTIONS**
        - List **essential precautions** the patient must follow.
        - Categorize by **priority levels** (High, Medium, Low).
        - Provide **clear, actionable recommendations** in **short, specific steps**.

        ## **3️⃣ SPECIALIST RECOMMENDATIONS**
        - Identify the **specific type of doctor(s)** the patient should consult.
        - Explain **why each specialist is recommended**.
        - Indicate if the consultation is **urgent or routine**.

        ## **4️⃣ PREDICTIONS**
        - Determine **possible conditions the patient may have**.
        - Predict **health outlook** based on the current report.
        - Highlight any **potential complications to watch for**.
        - Where applicable, give an estimated **timeframe for recovery**.

        ---

        Medical report content to analyze:
        {report}

        ### **⚠️ IMPORTANT INSTRUCTIONS FOR OUTPUT FORMAT:**
        ✅ **Use Markdown Formatting** for section titles (**bold headers**, bullet points).  
        ✅ **Ensure JSON output follows this structure exactly:**  
        ✅ **Provide **only** the json as requested. Do not include any explanations, comments, docstrings, or example usage.**

        ```json
        {{
            "DetailedAnalysis": "Patient exhibits signs of moderate hypertension with elevated blood pressure levels. No immediate life-threatening risks detected.",
            "Precautions": [
                {{
                    "precaution": "Monitor blood pressure daily and record readings.",
                    "priority": "High"
                }},
                {{
                    "precaution": "Reduce sodium intake and maintain a balanced diet.",
                    "priority": "Medium"
                }}
            ],
            "TypeOfDoctors": [
                {{
                    "specialist": "Cardiologist",
                    "reason": "Patient shows signs of hypertension and needs further cardiovascular evaluation.",
                    "urgency": "High",
                    "confidence": 95
                }}
            ],
            "Predictions": [
                {{
                    "prediction": "Patient is at risk of developing chronic hypertension if lifestyle changes are not implemented.",
                    "timeframe": "6 months",
                    "confidence": 80
                }},
                {{
                    "prediction": "With proper medication and lifestyle changes, condition may improve within 3 months.",
                    "timeframe": "3 months",
                    "confidence": 90
                }}
            ]
        }}
        ```
        

    '''

    # Generate content using Gemini
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    try:
        analysis_model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp", generation_config=generation_config)
        response = analysis_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"