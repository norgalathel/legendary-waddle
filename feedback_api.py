from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import feedback_generator

app = FastAPI()

# --- Input Models (from gidenveriler.tsx) ---
class DrivingDataPoint(BaseModel):
    timestamp: float
    speed: float
    acceleration: float
    latitude: Optional[float]
    longitude: Optional[float]
    elevation: Optional[float] = None
    # Accept any extra fields
    class Config:
        extra = "allow"

class SessionDetails(BaseModel):
    date: str
    consistency: float
    driver: str
    anonymous: bool
    trackName: Optional[str] = None
    lapNumber: Optional[int] = None

class NikiAIRequest(BaseModel):
    drivingData: List[DrivingDataPoint]
    sessionDetails: SessionDetails

# --- Output Models (from gelecekveri.tsx) ---
class NikiAISegment(BaseModel):
    id: int
    name: str
    startIndex: int
    endIndex: int
    feedback: str
    score: float
    type: str
    position: Dict[str, float]
    color: Optional[str] = None

class NikiAIPoint(BaseModel):
    id: int
    type: str
    position: Dict[str, float]
    feedback: str
    severity: str
    value: Optional[float] = None

class NikiAIPerformanceRegion(BaseModel):
    id: int
    center: Dict[str, float]
    radius: float
    type: str
    feedback: str

class NikiAIFeedback(BaseModel):
    segments: List[NikiAISegment]
    points: List[NikiAIPoint]
    performanceRegions: List[NikiAIPerformanceRegion]
    overallScore: float
    recommendations: List[str]
    drivingStyle: str
    efficiencyScore: float
    safetyScore: float
    currentSegment: Optional[NikiAISegment] = None

# --- Mapping function ---
def feedback_to_nikiaifeedback(feedback_dict):
    # Map feedback_dict to NikiAIFeedback
    segments = []
    points = []
    performanceRegions = []
    # Map corner feedbacks to segments
    seg_id = 1
    for fb in feedback_dict['feedback']:
        if fb.get('type') == 'corner':
            segments.append(NikiAISegment(
                id=seg_id,
                name=f"Corner {fb.get('corner', seg_id)}",
                startIndex=fb.get('corner', seg_id),
                endIndex=fb.get('corner', seg_id),
                feedback=" ".join(fb.get('feedback', [])),
                score=7.5,  # Placeholder, can be improved
                type='cornering',
                position={'lat': fb['latitude'], 'lng': fb['longitude']},
                color=None
            ))
            seg_id += 1
        elif fb.get('type') in ('positive', 'constructive'):
            # Map best/worst points to NikiAIPoint
            points.append(NikiAIPoint(
                id=len(points)+1,
                type='optimalLine' if fb['type']=='positive' else 'dangerZone',
                position={'lat': fb['latitude'], 'lng': fb['longitude']},
                feedback=" ".join(fb.get('feedback', [])),
                severity='low' if fb['type']=='positive' else 'high',
                value=None
            ))
    # Example: No performance regions for now
    # Map scores
    overallScore = round(feedback_dict.get('score', 80) / 10, 1)  # 0-10 scale
    recommendations = [feedback_dict.get('summary', '')]
    drivingStyle = "Balanced"  # Placeholder
    efficiencyScore = overallScore
    safetyScore = overallScore
    return NikiAIFeedback(
        segments=segments,
        points=points,
        performanceRegions=performanceRegions,
        overallScore=overallScore,
        recommendations=recommendations,
        drivingStyle=drivingStyle,
        efficiencyScore=efficiencyScore,
        safetyScore=safetyScore,
        currentSegment=None
    )

@app.post("/feedback", response_model=NikiAIFeedback)
def feedback_endpoint(request: NikiAIRequest):
    # Convert drivingData to DataFrame
    df = pd.DataFrame([d.dict() for d in request.drivingData])
    # For demo, use the same data as both optimal and real
    feedback_dict = feedback_generator.generate_feedback(df, df)
    return feedback_to_nikiaifeedback(feedback_dict) 