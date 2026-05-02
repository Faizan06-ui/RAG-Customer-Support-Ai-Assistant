import sys
import os

# Add the project root to the system path so we can import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Pipeline

if __name__ == "__main__":
    query = "How can I cancel my subscription?"
    rag = Pipeline()
    response = rag.run(query)
    print("\nFinal Response from AI:", response)
