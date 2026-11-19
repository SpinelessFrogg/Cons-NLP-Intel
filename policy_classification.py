
from typing import Literal
from pydantic import BaseModel, Field

class PolicyClassification(BaseModel):
    # restrict LLM output to only these mechanisms/sectors/alignment
    policy_mechanism: Literal[
        "Protected Area / Wildlife Corridor",
        "Conservation Funding & Appropriations",
        "Endangered Species & Harvest Rules",
        "Invasive Species Control",
        "Agency Administration / Program Reform",
        "Other"
    ]
    target_sector: Literal[
        "Wildlife & Biodiversity",
        "Wetlands & Watersheds",
        "Forests & Public Lands",
        "Coastal & Marine",
        "Agricultural / Working Lands",
        "Other"
    ]
    advocacy_alignment: Literal[
        "Strict Regulation",
        "Industry Relief / Exemption",
        "Neutral / Procedural"
    ]
    plain_english_summary: str = Field(
        description="One-sentence summary of what the bill actually does.",
        min_length=5
)