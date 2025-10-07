import os
import asyncio
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
import pandas as pd

load_dotenv()


async def research_single_question(
    ticker: str,
    question: str,
    question_num: int,
    per_point_index: int,
    point: int | None,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5
) -> tuple[int, str]:
    """
        Research a single question for a given ticker asynchronously with rate limiting and retry logic.
    
    Args:
        ticker (str): The stock ticker symbol.
        question (str): The question to research.
        question_num (int): The question number for tracking.
        per_point_index (int): Question number relative to current point group.
        point (int): The point group of this question.
        semaphore (asyncio.Sephamore): Semaphore to limit concurrent requests to avoid OpenAI rate limits.
        max_retries (int): Maximum number of retry attempts (default: 5)
    
    Returns:
        tuple[int, str]: Tuple containing the question number relative to all questions and the LLM's response.
    """
    
    # Make sure that the OpenAI API key is loaded.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    
    # Use async to make LLM calls in parallel, but use a sephamore to limit the number of LLM calls.
    async with semaphore: 
        # Print progress info to user about which question is being researched
        prefix = f"[Point {point} Q{per_point_index}]"
        print(f"{prefix} Starting research: {question[:80]}...") # Use a character limit to keep things compact
        
        client = AsyncOpenAI() # Initialize Async OpenAI client

        
        # Retry loop with exponential backoff in case a rate limit is hit
        for attempt in range(max_retries):
            try:
                # Create the request using GPT-5-mini with async.
                # GPT-5-mini is used because GPT-5 is too expensive to
                # be practical for independent use for most users.
                response = await client.responses.create(
                    model="gpt-5-mini",
                    input=f"""Research to find information which would aid in answering the question about the company associated with the ticker {ticker}. The question is: {question} Do not answer the question, simply provide the relevant information, and do not provide any commentary in the final output other than the facts. To convey the facts, use simple language and minimal jargon; explain the fact as if you are explaining to someone unfamiliar with the field/industry.""",
                    reasoning={"effort": "medium"},
                    text={"verbosity": "medium"},
                    tools=[{"type": "web_search"}]
                )
                
                # Update user on progress and return the text information corresponding with the current question.
                print(f"{prefix} Completed")
                return (question_num, response.output_text)
            except RateLimitError as e:
                
                wait_time = 2 ** (attempt + 1)
                
                # Wait and retry the request to OpenAI. After max_retries attempts, give up.
                if attempt < max_retries - 1:
                    print(f"{prefix} Rate limit hit. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"{prefix} Failed after {max_retries} attempts")
                    raise
        

async def research_multiple_questions(ticker: str, questions_df: pd.DataFrame, max_concurrent: int = 5) -> dict[int, str]:
    """
    Research multiple questions.


    Args:
        ticker (str): The stock ticker symbol.
        questions_df (pd.DataFrame): The dataframe storing all questions from questions.csv.
        max_concurrent (int, optional): The maximum amount of concurrent requests allowed. Defaults to 5.

    Returns:
        dict[int, str]: A dictionary holding all the responses to all the questions for which research was conducted.
    """
    # Print update message to the user
    print(f"Starting research for {len(questions_df)} questions...")
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Dictionary to keep track of how many questions have been
    # seen so far for each point.
    per_point_counters: dict[int, int] = {}

    research = []
    # Loop through the dataframe.
    # Assign the row index to question_num and a series containing the row data to row.
    for question_num, row in questions_df.iterrows():
        point_val = row['point']
        # Look up the current count value for this point. If it's not there yet, use 0.
        # Increment the count by 1.
        per_point_counters[point_val] = per_point_counters.get(point_val, 0) + 1
        per_point_index = per_point_counters[point_val]

        # Research the question corresponding with the point and question number.
        research_task = research_single_question(
            ticker=ticker,
            question=row['question'],
            question_num=question_num + 1,
            per_point_index=per_point_index,
            point=point_val,
            semaphore=semaphore
        )

        research.append(research_task)
    
    # Run all research tasks in parallel, but limited by the semaphore.
    results = await asyncio.gather(*research)
    
    # Convert the results to a dictionary
    results_dict = {num: result for num, result in results}
    
    return results_dict # Return a dictionary of the research results.



def save_to_markdown(
    ticker: str,
    content: str,
    point_value: int | str,
    evaluation_summary: str,
    filename: str | None = None, # Can be str or None; default value is None.
) -> str:
    """Save research content to a Markdown file.


    Args:
        ticker (str): The stock ticker symbol.
        content (str): The research content to save.
        point_value (int | str): The point value being researched.
        evaluation_summary (str): The evaluation of the evaluator model.
        filename (str | None, optional): Optional custom filename. Defaults to None.

    Returns:
        str: The full filename of the final Markdown file.
    """
    # Format the name of the final Markdown file to be outputted.
    if filename is None:
        date_str = datetime.now().strftime("%Y%m%d")
        if point_value != 'all':
            filename = f"reports/{ticker}_point{point_value}_{date_str}.md"
        else:
            filename = f"reports/{ticker}_all_{date_str}.md"

    
    # Create the title for the Markdown file
    if point_value != 'all':
        title_line = f"# Investment Research - Point {point_value}\n\n"
    else:
        title_line = f"# Investment Research - All Points\n\n"

    # Create a string which will eventually be pasted into the final Markdown file and add informatory text
    md_content = title_line
    md_content += f"**Company Ticker:** {ticker}  \n"
    md_content += f"**Date:** {datetime.now().strftime('%B %d, %Y')}  \n"
    if point_value and point_value != 'all':
        md_content += f"**Point:** {point_value}\n\n"
    else:
        md_content += "\n"

    # Insert evaluation summary
    md_content += evaluation_summary + "\n\n"

    # Insert more formatting and all the rest of the text for the file
    md_content += "---\n\n"
    md_content += content
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return filename

