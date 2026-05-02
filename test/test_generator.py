import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from rag.generator import Generator

# Create a generator instance
gen = Generator()

# User query
query = "I ordered a laptop, but it arrived with a broken screen. What should I do?"

# Dummy top matches with score (must be 3 elements per item!)
top_matches = [
    ("How do I cancel?", "You can cancel anytime via the settings.", 0.12),
    ("I want to stop my plan", "Go to billing → cancel plan", 0.23),
    ("Need to stop service", "Click 'Cancel subscription' in dashboard.", 0.31)
]

# Generate response
response = gen.generate_response(query, top_matches)

# Output
print("📦 AI Response:\n", response)

