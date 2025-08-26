import argparse
from rag.llm_rag import LLMRAGPipeline

def main():
    # Initialize the CLI parser
    parser = argparse.ArgumentParser(
        description="Ask a question and get an AI-generated answer using the LLMRAGPipeline."
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question you want to ask."
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5)."
    )
    parser.add_argument(
        "--filter_metadata",
        type=str,
        default=None,
        help="Optional metadata filter for document search (JSON string)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the LLM temperature (0.0-1.0)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens in the response."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the LLMRAGPipeline
    try:
        llm_rag = LLMRAGPipeline()  # Adjust if you need to pass specific configurations
    except Exception as e:
        print(f"Error initializing LLMRAGPipeline: {e}")
        return

    # Prepare LLM overrides
    llm_overrides = {}
    if args.temperature is not None:
        llm_overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        llm_overrides["max_tokens"] = args.max_tokens

    # Ask the question
    try:
        answer = llm_rag.answer(
            query=args.question,
            k=args.k,
            filter_metadata=args.filter_metadata,
            llm_overrides=llm_overrides if llm_overrides else None
        )

        # Display the result
        print("\n=== Answer ===")
        print(answer.answer)
        print("\n=== Sources ===")
        for source in answer.sources:
            print(f"- {source}")

    except Exception as e:
        print(f"Error answering question: {e}")

if __name__ == "__main__":
    main()