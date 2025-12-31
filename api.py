from fastapi import FastAPI, UploadFile, File, Depends , HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
import os
from sqlalchemy import (
    create_engine, Column, String, Float, Text, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
from starlette.responses import FileResponse
from main import chatbot_answer, extract_resume, extract_text_from_pdf,  calculate_experience_from_dates
from chromadb import PersistentClient
from main import get_embedding_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# LOAD ENV

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:root@localhost:5432/Resume-analyzer"
)


# DATABASE SETUP

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# DATABASE MODEL

class ResumeDB(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, index=True)
    email = Column(String,unique=True, index=True)
    phone = Column(String)
    skills = Column(Text)
    experience_years = Column(Float)
    experience_summary = Column(Text)
    education = Column(Text)
    projects = Column(Text)
    resume_link = Column(Text)
    source_file = Column(String)
    vector_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# PYDANTIC MODEL

class ResumeSchema(BaseModel):
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience_years: float = 0.0
    experience_summary: str = ""
    education: List[str] = []
    projects: List[str] = []
    resume_link: Optional[str] = None
    source_file: Optional[str] = None




class ResumeEmbedding:
    def __init__(self):
        os.makedirs("./chromadb", exist_ok=True)
        self.chromaclient = PersistentClient(path="./chromadb")
        
        # FIXED: NO MORE DELETE!
        try:
            self.collection = self.chromaclient.get_collection("resumes")
            print(" Found existing 'resumes' collection")
        except:
            self.collection = self.chromaclient.create_collection(
                name="resumes", 
                metadata={"hnsw:space": "cosine"}
            )
            print("Created NEW 'resumes' collection")
        
        self.embedder = get_embedding_model()
        print(" Embedding manager READY!")
        
embedding_manager = ResumeEmbedding()


# FASTAPI APP

app = FastAPI(title="Resume Analyzer")
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# APIs

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_resume(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    results = []

    for file in files:
        try:
            # Read file
            file_bytes = await file.read()

            # Save temporarily
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            # Extract text
            resume_text = extract_text_from_pdf(temp_path)
            os.remove(temp_path)

            # Run LLM extraction
            extracted = extract_resume(resume_text)
            cleaned = ResumeSchema(**extracted)
            # Fallback experience calculation
            if cleaned.experience_years == 0.0:
             # Try numeric years first
             years =  calculate_experience_from_dates(cleaned.experience_summary)
         
             if years > 0:
                 cleaned.experience_years = years
             else:
                 # Fallback to date-based calculation
                 cleaned.experience_years =  calculate_experience_from_dates(
                     cleaned.experience_summary
                 )
         

            existing = None
            if cleaned.email:
                existing = db.query(ResumeDB).filter(ResumeDB.email == cleaned.email).first()
            if not existing and cleaned.phone:
                existing = db.query(ResumeDB).filter(ResumeDB.phone == cleaned.phone).first()
            
            if existing:
                results.append({
                    "status": "duplicate",
                    "file": file.filename,
                    "message": f"Candidate already exists: {existing.name}",
                    "existing_id": str(existing.id)
                })
                continue

            # Save original PDF permanently
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            new_name = f"{uuid.uuid4()}_{file.filename}"
            final_path = os.path.join(UPLOAD_DIR, new_name)

            with open(final_path, "wb") as f:
                f.write(file_bytes)

            resume_link = f"http://127.0.0.1:8000/files/{new_name}"
            resume_text_for_embedding = f"""
              Name: {cleaned.name}
              Email: {cleaned.email}
              Phone: {cleaned.phone}
              Skills: {', '.join(cleaned.skills)}
              Experience: {cleaned.experience_years} years
              Summary: {cleaned.experience_summary}
              Education: {', '.join(cleaned.education)}
              Projects: {', '.join(cleaned.projects)}
              """.strip()
              
              # Generate embedding
            vec = embedding_manager.embedder.encode(resume_text_for_embedding)  # shape (384,) or (1,384)
            if len(vec.shape) == 2:
                 vec = vec[0]
            embedding = vec.tolist()

            vector_id = str(uuid.uuid4())
              
             # Save to ChromaDB
            embedding_manager.collection.add(
            embeddings=[embedding],
            documents=[resume_text_for_embedding],
            metadatas=[{"vector_id": vector_id, "file": new_name, "name": cleaned.name}],
            ids=[vector_id]
             )



            # Save to database
            db_record = ResumeDB(
                name=cleaned.name,
                email=cleaned.email,
                phone=cleaned.phone,
                skills=", ".join(cleaned.skills),
                experience_years=cleaned.experience_years,
                experience_summary=cleaned.experience_summary,
                education=", ".join(cleaned.education),
                projects=", ".join(cleaned.projects),
                resume_link=resume_link,
                source_file=new_name,
                vector_id=vector_id 
            )

          

            db.add(db_record)
            db.commit()

            results.append({
                "status": "success",
                "candidate_name": cleaned.name,
                "resume_link": resume_link
            })

        except Exception as e:
            db.rollback()
            results.append({"status": "error", "file": file.filename, "message": str(e)})

    return results




@app.post("/chat")
def chat(payload: dict, db: Session = Depends(get_db)):
    #print("SQL COUNT:", db.query(ResumeDB).count())
    #print("VECTOR COUNT:", embedding_manager.collection.count())

    question = payload.get("question", "").strip()
    if not question:
        return {"answer": "Please ask a question."}

    # 1️ Vector search - encode question first
    question_embedding = embedding_manager.embedder.encode(question)
    if len(question_embedding.shape) == 2:
        question_embedding = question_embedding[0]
    question_embedding = question_embedding.tolist()
    
    results = embedding_manager.collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )
    print("RAW VECTOR RESULTS:", results)

    if not results["metadatas"] or not results["metadatas"][0]:
        return {"answer": "No relevant resumes found."}

    # 2️ Fetch resumes from SQL
    resumes = []
    for meta in results["metadatas"][0]:
        vector_id = meta.get("vector_id")
        if not vector_id:
            continue

        resume = db.query(ResumeDB).filter(
            ResumeDB.vector_id == vector_id
        ).first()
        print("SQL RESUME FOUND:", resume)

        if resume:
            resumes.append(resume)

    if not resumes:
        return {"answer": "No matching resumes found in database."}

    # 3️ Build CONTEXT
    context = ""
    for i, r in enumerate(resumes):
        context += f"""
Candidate {i+1}
Name: {r.name}
Email: {r.email}
Phone: {r.phone}
Skills: {r.skills}
Experience: {r.experience_years} years
Summary: {r.experience_summary}
Education: {r.education}
Projects: {r.projects}
Resume Link: {r.resume_link}
---
"""

    # 4️ Ask LLM
    answer = chatbot_answer(question, context)

    return {
        "answer": answer,
        "best_matches": [
            {
                "name": r.name,
                "skills": r.skills,
                "experience": r.experience_years,
                "resume_link": r.resume_link
            }
            for r in resumes
        ]
    }


    
#  Resume download
@app.get("/resume/{resume_id}")
async def get_resume(resume_id: str, db: Session = Depends(get_db)):
    resume = db.query(ResumeDB).filter(ResumeDB.id == resume_id).first()
    if resume:
        file_path = os.path.join(UPLOAD_DIR, resume.source_file)
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                filename=resume.source_file,
                media_type="application/pdf"
            )
    raise HTTPException(404, "Resume not found")


@app.get("/files/{filename}")
def serve_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    print("Trying to load:", file_path)   

    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}

    return FileResponse(path=file_path, media_type="application/pdf")


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total_resumes = db.query(ResumeDB).count()
    return {"total_resumes": total_resumes}

@app.get("/vector-stats")
def vector_stats():
    try:
        count = embedding_manager.collection.count()
        return {"total_vectors": count}
    except:
        return {"total_vectors": 0}

@app.get("/debug")
def debug(db: Session = Depends(get_db)):
    sql_count = db.query(ResumeDB).count()
    try:
        vector_count = embedding_manager.collection.count()
        samples = embedding_manager.collection.get(limit=3)
    except:
        vector_count, samples = 0, {"documents": []}
    return {
        "sql_resumes": sql_count,
        "vector_count": vector_count,
        "sample_embeddings": [d[:100] for d in samples.get('documents', [])]
    }


# TEST embedding on startup
@app.on_event("startup")
async def startup_event():
    try:
        print(" Testing embedding model...")
        model = get_embedding_model()
        test_vec = model.encode(["test"])
        print(f" Embedding test OK! Shape: {test_vec.shape}")
    except Exception as e:
        print(f" STARTUP EMBEDDING FAILED: {str(e)}")
@app.get("/debug-vectorids")
def debug_vectorids(db: Session = Depends(get_db)):
    resumes = db.query(ResumeDB).filter(ResumeDB.vector_id != None).all()
    vectors = embedding_manager.collection.get()
    return {
        "sql_vectorids": [r.vector_id for r in resumes[:3]],
        "chroma_ids": vectors.get('ids', [])[:3],
        "chroma_vectorid_metadata": [m.get('vector_id') for m in vectors.get('metadatas', [])[:3]]
    }

