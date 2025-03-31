from predict_d import predict_d
from predict_c import predict_c
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from io import BytesIO
from bson import ObjectId
from src.google_search_tools import GoogleSearchTools
from pydantic import BaseModel
from src.crawl import Crawl4aiTools
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

class Id(BaseModel):
    obj_id: str


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
def classify_image(req: Id):
    obj_id = req.obj_id
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
                        print(image_url)  
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
            tools=[DuckDuckGoTools(),GoogleSearchTools(),Crawl4aiTools()],
            markdown=True,
        )

        result: RunResponse = verify_med_agent.run("Please analyze this medical image.", images=[agno_image])
        result_d = predict_d(image)
        result_c = predict_c(image)

        if result_c["confidence"] > result_d["confidence"]:
            result_pred = result_c
            minor_result = result_d
        elif result_d["confidence"] > result_c["confidence"]:
            result_pred = result_d
            minor_result = result_c
        else:
            result_pred = result_c        
        unhealthy_skin_agent = Agent(
            name="Medical Imaging Analysis Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools(),predict_c(image),predict_d(image)],
            context={"verify": result.content},
            add_context=True,
            markdown=True,

        )
        pred: RunResponse = unhealthy_skin_agent.run("Please analyze this medical image.", images=[agno_image])
        diag_collection = db["diag"] 
        diag_collection.replace_one({}, {"pred": pred.content}, upsert=True)

        report_agent = Agent(
            name="Medical Imaging Analysis and report generator Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools(),predict_c(image),predict_d(image),GoogleSearchTools()],
            context={"pred": pred.content},
            add_context=True,
            markdown=True,
            instructions=dedent(f"""# Skin Disease Diagnosis Report  
                                If the skin classification is unhealthy then in report also add the our model predicted that  {pred.content}  but it also give two answer                         
## Step 1: Image Technical Assessment  

### 1.1 Imaging & Quality Review  
- Imaging Modality Identification: (Dermatoscopic, Clinical, Histopathological, etc.)  
- Anatomical Region & Patient Positioning: (Specify if available)  
- Image Quality Evaluation: (Contrast, Clarity, Presence of Artifacts)  
- Technical Adequacy for Diagnostic Purposes: (Yes/No, with reasoning)  

### 1.2 Professional Dermatological Analysis  
- Systematic Anatomical Review  
- Primary Findings: (Lesion Size, Shape, Texture, Color, etc.)  
- Secondary Observations (if applicable)  
- Anatomical Variants or Incidental Findings  
- Severity Assessment: (Normal / Mild / Moderate / Severe)  

---

## Step 2: Context-Specific Diagnosis & Clinical Interpretation  
- Primary Diagnosis: (Detailed interpretation based on the given disease context)  
- Secondary Condition (if suspected): (Mention briefly without shifting focus)  

---

## Step 3: Recommended Next Steps  
- Home Remedies & Skincare: (Moisturizing, Avoiding Triggers, Hydration)  
- Medications & Treatments: (Antifungal, Antibiotic, Steroid Creams, Oral Medications)  
- When to See a Doctor: (Persistent Symptoms, Spreading, Bleeding, Painful Lesions)  
- Diagnostic Tests (if required): (Skin Biopsy, Allergy Tests, Blood Tests)  

---

## Step 4: Patient Education  
- Clear, Jargon-Free Explanation of Findings  
- Visual Analogies & Simple Diagrams (if helpful)  
- Common Questions Addressed  
- Lifestyle Implications (if any)  

---

## Step 5: Ayurvedic or Home Solutions  
(Applied only if the condition is non-cancerous or mild and use web search)  
- Dry & Irritated Skin: Apply Aloe Vera gel, **Coconut oil, or **Ghee for deep moisturization.  
- Inflammation & Redness: Use a paste of Sandalwood (Chandan) and Rose water for cooling effects.  
- Fungal & Bacterial Infections: Apply Turmeric (Haldi) paste with honey or Neem leaves for antimicrobial benefits.  
- Eczema & Psoriasis: Drink Giloy (Guduchi) juice and use a paste of Manjistha & Licorice (Yashtimadhu) for skin detox.  

---

## Step 6: Evidence-Based Context & References  
Using DuckDuckGo search and provide relevent links:  
- Recent relevant medical literature  
- Standard treatment guidelines  
- Similar case studies  
- Technological advances in imaging/treatment  
- 2-3 authoritative medical references
- give related links also with references.  

---

## Final Summary & Conclusion  
Key Takeaways:  
- Most Likely Diagnosis: (Brief summary)  
- Recommended Actions: (Main steps for treatment and next consultation)  
The most likely condition the patient could have is **{result_pred['class']}** with a confidence of {result_pred['confidence']:.2f}. 
Additionally, there is a minor possibility of **{minor_result['class']}** with a confidence of {minor_result['confidence']:.2f}. 

**Remarks:**  
- **{result_pred['class']}** (Confidence: {result_pred['confidence']:.2f}) is the primary concern and should be prioritized for diagnosis and treatment.  
- **{minor_result['class']}** (Confidence: {minor_result['confidence']:.2f}) may be a secondary condition or share similar symptoms. Further medical evaluation is recommended to rule it out.
Note: This report is AI-generated and should not replace professional medical consultation. Always consult a dermatologist for a confirmed diagnosis and personalized treatment.  
 - give answer in proper markdown format.

---
""")
        )
        report: RunResponse = report_agent.run("Please analyze this skin image output context and generate a proper report for Dermatologist to understand.", images=[agno_image])
        net_agent = Agent(
            name="Medical Imaging Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[DuckDuckGoTools(),Crawl4aiTools()],  
            context={"pred": pred.content},
            add_context=True,
            markdown=True,  
            instructions=dedent(
                f"""
                **Remarks:**  
                - **{result_pred['class']}** (Confidence: {result_pred['confidence']:.2f}) is the primary concern and should be prioritized for diagnosis and treatment.  
                - **{minor_result['class']}** (Confidence: {minor_result['confidence']:.2f}) may be a secondary condition or share similar symptoms. Further medical evaluation is recommended to rule it out.

                """)

        )
        jarvis: RunResponse = net_agent.run("Please analyze this skin based diagnostics report and give instructions to doctor")
        return {"image_url": image_url, "verify": result.content, "prediction": pred.content,"report":report.content,"jarvis": jarvis.content}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
class Query(BaseModel):
    query: str
    deep_search: bool = False

@app.post("/ans")
def get_ans(q: Query):
    try:
        client=MongoClient(MONGO_URI)
        db = client["test"]  
        diag_collection = db["diag"] 
        latest_entry = diag_collection.find_one({}, {"_id": 0})
        mongo_pred = latest_entry["pred"]
    except Exception as e:
        w_agent = Agent(
                name="Skin Diesease Research Expert",
                model=Groq(id="qwen-2.5-32b"),
                tools=[DuckDuckGoTools()],  
                markdown=True,  
            )
        get_w_result: RunResponse = w_agent.run(q.query)
        return {"response": get_w_result.content}
    web_agent = Agent(
            name="Skin Diesease Research Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[DuckDuckGoTools()],  
            context={"diagnosis": mongo_pred},
            add_context=True,
            markdown=True,  
        )
    deep_agent = Agent(
            name="Skin Disease Research Expert",
            model=Groq(id="qwen-2.5-32b"),
            tools=[ArxivTools()],  
            context={"pred_diagnostic": mongo_pred},
            add_context=True,
            markdown=True,  
            )
    if q.deep_search:
        get_deep_result: RunResponse = deep_agent.run(q.query)
        return {"response": get_deep_result.content}
    else:
        get_web_result: RunResponse = web_agent.run(q.query)
        return {"response": get_web_result.content}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=6700, reload=True)