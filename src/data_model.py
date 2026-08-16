from pydantic import BaseModel

"""
Structured data models used by the top-down and bottom-up goal pipeline.

This module contains only the base/original models shared by the whole
pipeline. Models and Enums introduced specifically for the bottom-up
feedback-loop extension (bottom-up HLG reconstruction, the Global Goal
Evaluator, documentation coverage, and the outer refinement-abstraction-
verification cycle) live in ``src.bottom_up.models``.
"""


class DocumentDescription(BaseModel):
    description: str


class Actor(BaseModel):
    name: str
    description: str


class Actors(BaseModel):
    actors: list[Actor]


class HighLevelGoal(BaseModel):
    name: str
    description: str
    actor: Actor


class HighLevelGoals(BaseModel):
    goals: list[HighLevelGoal]


class LowLevelGoal(BaseModel):
    name: str
    description: str
    high_level_associated: HighLevelGoal


class LowLevelGoals(BaseModel):
    low_level_goals: list[LowLevelGoal]


class API(BaseModel):
    api_name: str
    api_path: str
    description: str
    request_type: str


class APIMapping(BaseModel):
    APIs: list[API]
    low_level_goal: LowLevelGoal


class Critique(BaseModel):
    score: float
    comment: str
