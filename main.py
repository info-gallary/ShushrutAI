from predict import predict
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from io import BytesIO
from bson import ObjectId


from dotenv import load_dotenv
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from agno.models.groq import Groq
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from pymongo import MongoClient

load_dotenv()

MONGO_URI=os.getenv("MONGO_URI")





app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Welcome to the ShrushrutAI"}

@app.post("/predict")
def classify_image(obj_id: str):
    try:
        client=MongoClient(MONGO_URI)
        db = client["test"]  
        collection = db["predicts"] 
        def get_latest_skin_image(obj_id: str):
            try:
                obj_id = ObjectId(obj_id)  
                document = collection.find_one({"patient": obj_id})  

                if document:
                    image_url = document.get("latestSkinImage")
                    if image_url:
                        print(image_url)  # Print the image URL
                        return image_url
                    else:
                        print("No image found in document")
                        return None
                else:
                    print("Document not found")
                    return None

            except Exception as e:
                print(f"Error: {e}")
                return None

        image_url = get_latest_skin_image(obj_id)
        if not image_url:
            raise HTTPException(status_code=404, detail="Image not found")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        image_bytes = img_bytes.getvalue()
        agno_image = AgnoImage(content=image_bytes, format="png")
        verify_med_agent = Agent(
            name="Medical Imaging Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools()],
            markdown=True,
            instructions=dedent(
                f"""Analyze the given skin image as an very good and expert dermatologist and expert to determine if the skin is **healthy or unhealthy**.
                    - confidence percentage should be between 90 to 100 and you can use the decimal value also.
                    - If healthy, classify it as 'Healthy' and provide the confidence level in percentage.
                    - If unhealthy, classify it as 'Unhealthy' and provide the confidence level in percentage.
                    - Additionally, determine the skin type as one of the following: 'Dry', 'Oily', or 'Normal'.
                    - give answer in strictly <classification>,<confidence score in percent>,<skin type>,<remarks> format only.
                    """
            )
        )

        result: RunResponse = verify_med_agent.run("Please analyze this medical image.", images=[agno_image])
        result_pred = predict(image)
        unhealthy_skin_agent = Agent(
            name="Medical Imaging Analysis Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools()],
            context={"verify": result.content},
            add_context=True,
            markdown=True,
            instructions=dedent(
                f"""Analyze the given skin image as an expert dermatologist.If the skin appears healthy, classify the prediction as it 'Healthy' and provide the confidence level. If unhealthy, use the model output to determine the disease. The prediction is by deep learning model is {result_pred}. If classified as one of the following: 'Actinic Keratosis', 'Atopic Dermatitis', 'Benign Keratosis', 'Dermatofibroma', 'Melanocytic Nevus', 'Melanoma', 'Squamous Cell Carcinoma', 'Tinea Ringworm Candidiasis', or 'Vascular Lesion', assess the likelihood of skin cancer. Provide the disease name, confidence level, and remarks. Additionally, include possible symptoms that might be present for further diagnostic evaluation.
                - give answer in strictly <disease>,<confidence score in percent>,<remarks in two to three lines> format only.
                - If the skin appears healthy, classify it as 'Healthy' and provide the confidence level in percentage.
                """
            )
        )
        pred: RunResponse = unhealthy_skin_agent.run("Please analyze this medical image.", images=[agno_image])
        diag_collection = db["diag"] 
        diag_collection.replace_one({}, {"pred": pred.content}, upsert=True)

        report_agent = Agent(
            name="Medical Imaging Analysis and report generator Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools()],
            context={"pred": pred.content},
            add_context=True,
            markdown=True,
            instructions=dedent("""# **Skin Disease Diagnosis Report** 🏥  

## **Step 1: Image Technical Assessment**  

### **1.1 Imaging & Quality Review**  
- **Imaging Modality Identification:** _(Dermatoscopic, Clinical, Histopathological, etc.)_  
- **Anatomical Region & Patient Positioning:** _(Specify if available)_  
- **Image Quality Evaluation:** _(Contrast, Clarity, Presence of Artifacts)_  
- **Technical Adequacy for Diagnostic Purposes:** _(Yes/No, with reasoning)_  

### **1.2 Professional Dermatological Analysis**  
- **Systematic Anatomical Review**  
- **Primary Findings:** _(Lesion Size, Shape, Texture, Color, etc.)_  
- **Secondary Observations (if applicable)**  
- **Anatomical Variants or Incidental Findings**  
- **Severity Assessment:** _(Normal / Mild / Moderate / Severe)_  

---

## **Step 2: Context-Specific Diagnosis & Clinical Interpretation**  
- **Primary Diagnosis:** _(Detailed interpretation based on the given disease context)_  
- **Secondary Condition (if suspected):** _(Mention briefly without shifting focus)_  

---

## **Step 3: Recommended Next Steps**  
- **Home Remedies & Skincare:** _(Moisturizing, Avoiding Triggers, Hydration)_  
- **Medications & Treatments:** _(Antifungal, Antibiotic, Steroid Creams, Oral Medications)_  
- **When to See a Doctor:** _(Persistent Symptoms, Spreading, Bleeding, Painful Lesions)_  
- **Diagnostic Tests (if required):** _(Skin Biopsy, Allergy Tests, Blood Tests)_  

---

## **Step 4: Patient Education**  
- **Clear, Jargon-Free Explanation of Findings**  
- **Visual Analogies & Simple Diagrams (if helpful)**  
- **Common Questions Addressed**  
- **Lifestyle Implications (if any)**  

---

## **Step 5: Ayurvedic or Home Solutions**  
_(Applied only if the condition is non-cancerous or mild and use web search)_  
- **Dry & Irritated Skin:** Apply **Aloe Vera gel**, **Coconut oil**, or **Ghee** for deep moisturization.  
- **Inflammation & Redness:** Use a paste of **Sandalwood (Chandan)** and **Rose water** for cooling effects.  
- **Fungal & Bacterial Infections:** Apply **Turmeric (Haldi) paste** with honey or **Neem leaves** for antimicrobial benefits.  
- **Eczema & Psoriasis:** Drink **Giloy (Guduchi) juice** and use a paste of **Manjistha & Licorice (Yashtimadhu)** for skin detox.  

---

## **Step 6: Evidence-Based Context & References**  
🔬 Using DuckDuckGo search and provide relevent links:  
- **Recent relevant medical literature**  
- **Standard treatment guidelines**  
- **Similar case studies**  
- **Technological advances in imaging/treatment**  
- **2-3 authoritative medical references**
- give related links also with references.  

---

## **Final Summary & Conclusion**  
📌 **Key Takeaways:**  
- **Most Likely Diagnosis:** _(Brief summary)_  
- **Recommended Actions:** _(Main steps for treatment and next consultation)_  

**Note:** This report is **AI-generated** and should **not** replace professional medical consultation. Always consult a **dermatologist** for a confirmed diagnosis and personalized treatment.  

---
""")
        )
        report: RunResponse = report_agent.run("Please analyze this skin image output context and generate a proper report for Dermatologist to understand.", images=[agno_image])
        net_agent = Agent(
            name="Medical Imaging Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[DuckDuckGoTools()],  
            context={"pred": pred.content},
            add_context=True,
            markdown=True,  
            instructions=dedent(
                f"""You are an **AI-powered Dermatology Voice Assistant**, designed to provide expert-level support to dermatologists. Your role is to **analyze report {report.content} recommend evidence-based treatments, and guide doctors on the next steps** using the latest research and drug discoveries.  

                When a doctor asks about a skin disease, follow these steps:  

                ### **1️⃣ Understand & Analyze the Case**  
                - Listen to the doctor’s query about a patient’s condition.  
                - Identify the disease or condition being discussed.  
                - Analyze symptoms, affected areas, and disease progression based on the given **context or medical report**.  

                ### **2️⃣ Provide the Latest Treatment Recommendations**  
                - Fetch **current treatment guidelines, FDA-approved drugs, and clinical trials** using web sources.  
                - Explain the **best available treatment options**, including **topical, oral, biologic, and advanced therapies**.  
                - Compare **traditional treatments** with **newly discovered therapies** (e.g., AI-assisted skin diagnostics, gene therapy, biologics).  

                ### **3️⃣ Generate a Complete Prescription Plan**  
                - Suggest **medications, dosages, frequency, and possible side effects**.  
                - Recommend **adjunct therapies**, such as lifestyle modifications and skincare routines.  
                - Warn about **contraindications or potential drug interactions**.  

                ### **4️⃣ Guide the Doctor on the Next Steps**  
                - Recommend **further diagnostic tests** (e.g., biopsy, dermoscopy, blood tests, genetic markers).  
                - Suggest **patient follow-up intervals and monitoring plans**.  
                - Provide **guidelines for managing severe or resistant cases**.  

                ### **5️⃣ Provide Reliable Medical Sources & Links**  
                - Fetch **research-backed insights** from trusted sources such as **PubMed, JAMA Dermatology, The Lancet, FDA, and WHO**.  
                - Offer **links to the latest studies, treatment guidelines, and clinical trials** for validation.  

                ---

                ### **🔊 Voice Assistant Example Responses**  

                💬 **Doctor:** "What are the latest treatments for atopic dermatitis?"  
                🔊 **AI Response:**  
                ✅ "Recent research suggests that **Dupilumab (IL-4/IL-13 inhibitor)** and **JAK inhibitors (Upadacitinib, Abrocitinib)** are highly effective for moderate to severe cases. Would you like a prescription recommendation?"  

                💬 **Doctor:** "Yes, suggest a treatment plan."  
                🔊 **AI Response:**  
                ✅ "For moderate atopic dermatitis, consider:  
                - **Topical:** Tacrolimus 0.1% (Apply twice daily)  
                - **Oral:** Upadacitinib 15 mg (Once daily)  
                - **Biologic:** Dupilumab 600 mg (Initial dose), then 300 mg every two weeks  
                - **Adjunct:** Barrier repair moisturizer and avoidance of known triggers  
                Would you like to know when to follow up with the patient?"  

                💬 **Doctor:** "Yes, what are the next steps?"  
                🔊 **AI Response:**  
                ✅ "Monitor for **infection signs, side effects, and disease progression**. If no improvement in **4 weeks**, consider patch testing for allergens or adding phototherapy. Here’s a **link to the latest clinical guidelines**: [FDA Dermatology Treatments](https://www.fda.gov/drugs)."  

                ---

                <Instructions should be understandable by Dermatologists and should not be too technical and make it like a proffesional doctor is giving advice to the other doctor and make complete instruction summarize and in 4 to 5 lines pointwise.>  

                """)

        )
        jarvis: RunResponse = net_agent.run("Please analyze this skin based diagnostics report and give instructions to doctor")
        return {"image_url": image_url, "verify": result.content, "prediction": pred.content,"report":report.content,"jarvis": jarvis.content}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

@app.post("/ans")
def get_ans(question: str,deep_search: bool = False):
    client=MongoClient(MONGO_URI)
    db = client["test"]  
    diag_collection = db["diag"] 
    latest_entry = diag_collection.find_one({}, {"_id": 0})
    mongo_pred = latest_entry["pred"]
    web_agent = Agent(
            name="Skin Diesease Research Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[DuckDuckGoTools()],  
            context={"diagnosis": mongo_pred},
            add_context=True,
            markdown=True,  
            instructions=dedent(
                f"""Analyze the given question as an expert dermatologist. use web to get more insights and make sure that your answers are based on doctor point of view that a dermatologist should understand that.
                 - give proper links and references.
                   """),

        )
    deep_agent = Agent(
            name="Skin Disease Research Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[ArxivTools()],  
            context={"pred_diagnostic": mongo_pred},
            add_context=True,
            markdown=True,  
            instructions=dedent(
                f"""Analyze the given question as an expert dermatologist. use Arxiv to get latest research and discoveries and make sure that your answers are based on doctor point of view that a dermatologist should understand that. and that provide the new treatment discoveries and latest research on treatment and drug that is done on the disease and how to treat it.
                 - give proper links and references.
                 - if not context provided then just act as a professional Dermatologist and give the answer.
                   """),

        )
    if deep_search:
        get_deep_result: RunResponse = deep_agent.run(question)
        return {"response": get_deep_result.content}
    else:
        get_web_result: RunResponse = web_agent.run(question)
        return {"response": get_web_result.content}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=4501, reload=True)