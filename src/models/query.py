"""Query data structures and models"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Query:
    """Represents a single LLM query with responses"""
    query_id: int
    user_query: str
    llm_response_1: str
    llm_response_2: Optional[str]
    unix_timestamp: int
    
    def has_pairwise_responses(self) -> bool:
        """Check if this query has two responses for pairwise comparison"""
        return (self.llm_response_2 is not None and 
                self.llm_response_2.strip() not in ['NULL', '', 'null'])
