from typing import Any, Dict, List, Optional
import asyncio
import json

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.graph.connections import graph_interface, schema_provider  # noqa: F401
from src.graph.graph_interface import clear_query_log, drain_query_log
from src.workflow.graph_builder import build_workflow
from src.workflow.state import ConversationState, InputState, create_initial_state


load_dotenv()

# Build LangGraph workflow once at startup
workflow = build_workflow()

# Human-readable step descriptions
STEP_DESCRIPTIONS = {
    "entity_extractor": "Extracting entities from your question...",
    "entity_normalizer": "Normalizing entities to knowledge graph format...",
    "supervisor": "Planning next analysis step...",
    "safety_check": "Checking supplement-medication interactions...",
    "deficiency_check": "Analyzing potential nutrient deficiencies...",
    "recommendation": "Finding supplement recommendations...",
    "synthesis": "Synthesizing final answer...",
}


class PatientProfile(BaseModel):
    medications: str = ""
    supplements: str = ""
    conditions: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    user_question: str
    patient_profile: PatientProfile
    # Included for future session/memory support; not yet persisted
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    final_answer: Optional[str]
    safety_results: Optional[Dict[str, Any]] = None
    deficiency_results: Optional[Dict[str, Any]] = None
    recommendation_results: Optional[Dict[str, Any]] = None
    evidence_chain: List[str] = Field(default_factory=list)
    iterations: int = 0
    error_message: Optional[str] = None


app = FastAPI(title="Supplement Safety Advisor API")


# CORS – allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Create React App / Next dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Run a full LangGraph workflow execution for a single question + profile.

    This is currently stateless per request; session_id is reserved for
    future persistent memory / multi-turn conversations.
    """
    try:
        input_state: InputState = {
            "user_question": payload.user_question,
            "patient_profile": payload.patient_profile.model_dump(),
        }

        initial_state: ConversationState = create_initial_state(
            user_question=input_state["user_question"],
            patient_profile=input_state["patient_profile"],
        )

        result: ConversationState = workflow.invoke(initial_state)  # type: ignore[arg-type]

        return ChatResponse(
            final_answer=result.get("final_answer"),
            safety_results=result.get("safety_results"),
            deficiency_results=result.get("deficiency_results"),
            recommendation_results=result.get("recommendation_results"),
            evidence_chain=result.get("evidence_chain", []),
            iterations=result.get("iterations", 0),
            error_message=result.get("error_message"),
        )
    except Exception as exc:  # noqa: BLE001
        # For now, surface a generic error to the client and log server-side
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    """
    Stream LangGraph workflow execution with step-by-step progress updates.

    Returns Server-Sent Events (SSE) with:
    - step: Fired when a node starts (from debug "task" event)
    - step_update: Fired after supervisor finishes, carries the routing decision
    - result: Final result when complete
    - error: Error message if something goes wrong
    """
    async def generate():
        try:
            input_state: InputState = {
                "user_question": payload.user_question,
                "patient_profile": payload.patient_profile.model_dump(),
            }

            initial_state: ConversationState = create_initial_state(
                user_question=input_state["user_question"],
                patient_profile=input_state["patient_profile"],
            )

            completed_steps: List[str] = []
            final_state: Dict[str, Any] = dict(initial_state)
            all_queries: List[Dict[str, Any]] = []

            clear_query_log()

            for mode, event in workflow.stream(initial_state, stream_mode=["updates", "debug"]):

                if mode == "debug":
                    # Fire SSE as soon as a node starts
                    if event.get("type") == "task":
                        node_name = event.get("payload", {}).get("name", "")
                        if node_name and node_name in STEP_DESCRIPTIONS:
                            step_info: Dict[str, Any] = {
                                "type": "step",
                                "node": node_name,
                                "description": STEP_DESCRIPTIONS[node_name],
                                "completed_steps": completed_steps.copy(),
                            }
                            yield f"data: {json.dumps(step_info)}\n\n"
                            await asyncio.sleep(0)

                elif mode == "updates":
                    # Node finished — collect output and queries
                    for node_name, node_output in event.items():
                        final_state.update(node_output)

                        step_queries = drain_query_log()
                        all_queries.extend(
                            {**q, "node": node_name} for q in step_queries
                        )

                        # For supervisor, send a follow-up event with the routing decision
                        if node_name == "supervisor":
                            decision = node_output.get("supervisor_decision", "")
                            if decision:
                                decision_info: Dict[str, Any] = {
                                    "type": "step_update",
                                    "node": node_name,
                                    "decision": decision,
                                }
                                yield f"data: {json.dumps(decision_info)}\n\n"
                                await asyncio.sleep(0)

                        completed_steps.append(node_name)

            result_data = {
                "type": "result",
                "final_answer": final_state.get("final_answer"),
                "safety_results": final_state.get("safety_results"),
                "deficiency_results": final_state.get("deficiency_results"),
                "recommendation_results": final_state.get("recommendation_results"),
                "evidence_chain": final_state.get("evidence_chain", []),
                "iterations": final_state.get("iterations", 0),
                "all_queries": all_queries,
                "workflow_steps": completed_steps,
            }
            yield f"data: {json.dumps(result_data)}\n\n"
            await asyncio.sleep(0)

        except Exception as exc:
            error_data = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def run():
    """Convenience entrypoint for `pdm run api`."""
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()

