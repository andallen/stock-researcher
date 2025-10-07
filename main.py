import sys
import asyncio
import pandas as pd
import time
from researcher import research_multiple_questions, save_to_markdown
from evaluator import evaluate_points


async def main_async():

    # Print usage guide if less than 3 arguments are passed in
    if len(sys.argv) < 3:
        print("Usage: python main.py <stock_ticker|tickers.txt> <point_value|all> [max_concurrent]")
        print("  stock_ticker:   Single ticker symbol (e.g., AAPL, TSLA)")
        print("  tickers.txt:    Batch mode using symbols listed one per line")
        print("  point_value:    Integer 1-18 selecting a single qualitative point")
        print("  all:            Research all points/questions")
        print("  max_concurrent: Optional concurrency (default 5)")
        print("\nExamples:")
        print("  python main.py AAPL all              # All points for AAPL")
        print("  python main.py AAPL 7                # Point 7 only")
        print("  python main.py AAPL 7 3              # Point 7 only, concurrency=3")
        print("  python main.py tickers.txt all       # All points for all tickers")
        print("  python main.py tickers.txt all 10    # All points, concurrency=10")
        print("  python main.py tickers.txt 12        # Point 12 only for all tickers")
        print("  python main.py tickers.txt 12 6      # Point 12 only, concurrency=6")
        return

    ticker_input = sys.argv[1]
    
    tickers = []
    if ticker_input.lower() == 'tickers.txt': # If tickers.txt was passed in...
        try:
            with open('tickers.txt', 'r') as f: # Open the file
                tickers = [line.strip().upper() for line in f if line.strip()] # Format the tickers in the file
            print(f"Loaded {len(tickers)} tickers from tickers.txt: {', '.join(tickers)}") # Update the user
        except FileNotFoundError: # Handle the case in which the file isn't found
            print("Error: tickers.txt file not found")
            return
    else:
        tickers = [ticker_input.upper()] # If tickers.txt was not passed in, set tickers to equal the single ticker that was passed in.
    
    # Parse the mandatory point|all argument
    point_arg = sys.argv[2].lower()
    point_value = None  # None signifies 'all'
    if point_arg != 'all':
        try:
            point_value = int(point_arg)
        # Handle the case in which an integer isn't passed in for the argument
        except ValueError:
            print("Error: <point_value|all> must be 'all' or an integer 1-18")
            return
        # Handle the case in which an out-of-range integer is passed in
        if point_value < 1 or point_value > 18:
            print("Error: point_value must be between 1 and 18 (inclusive)")
            return

    # Handling the optional concurrency argument
    max_concurrent = 5
    if len(sys.argv) >= 4:
        # If the argument corresponding to max_concurrent was passed in,
        # set max_concurrent and make sure it is a valid value. Otherwise,
        # return.
        try:
            max_concurrent = int(sys.argv[3])
            if max_concurrent < 1:
                print("Error: max_concurrent must be at least 1")
                return
        except ValueError:
            print("Error: max_concurrent must be an integer")
            return

    # Process each ticker
    for ticker_idx, ticker in enumerate(tickers, 1):
        # If there is at least one ticker, print an update to the user about processing the ticker
        if len(tickers) > 1:
            print(f"\n{'='*60}")
            print(f"  Processing Ticker {ticker_idx}/{len(tickers)}: {ticker}")
            print(f"{'='*60}\n")

        # Print more information for the user
        print(f"\n{'='*60}")
        print(f"  Investment Research - {ticker}")

        if point_value is not None:
            print(f"  Point: {point_value}")
        else:
            print(f"  Point: All Questions")
        print(f"  Max Concurrent Requests: {max_concurrent}")
        print(f"{'='*60}\n")

        if point_value is not None:
            print(f"Loading questions for point {point_value}...")
        else:
            print(f"Loading all questions...")
        
        # Convert subquestions.csv into a Pandas dataframe
        df = pd.read_csv('questions/subquestions.csv')
        
        # Filter the dataframe to a single point if it was specified, or use all questions
        if point_value is not None:
            filtered_df = df[df['point'] == point_value].reset_index(drop=True)
            
            print(f"Loaded {len(filtered_df)} question(s) for point {point_value}")
        else:
            filtered_df = df.reset_index(drop=True)
            print(f"Loaded {len(filtered_df)} total question(s)\n")
        
        # Start timer
        print("Starting stopwatch to time research!")
        start_time = time.time()

        # Research all questions in parallel (with rate limiting)
        results = await research_multiple_questions(ticker, filtered_df, max_concurrent)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        
        # Print how long the research took
        print(f"\n{'='*60}")
        print(f"  Research Complete - Compiling Results")
        if minutes > 0:
            print(f"  Total Time: {minutes}m {int(seconds)}s")
        else:
            print(f"  Total Time: {int(seconds)}s")
        print(f"{'='*60}\n")
        
        # Create the output which will be written to the final markdown file
        if point_value is not None:
            combined_output = f"Investment Research for {ticker} - Point {point_value}\n"
        else:
            combined_output = f"Investment Research for {ticker} - All Questions\n"
        combined_output += f"{'='*60}\n\n"

        # Create a dictionary which stores key value pairs of 
        # the point and the question corresponding to that point.
        points_dict = {}
        points_df = pd.read_csv('questions/points.csv')
        points_dict = dict(zip(points_df['point'], points_df['question']))

        # Initialize dictionary to keep track of how many questions have
        # been processed for the current point.
        per_point_counters = {}
        prev_point = None # Variable to store the point of the previous question.
        for question_index, row in filtered_df.iterrows():
            curr_point = row['point'] # Variable to store the point of the current question.

            # If the current point is different from the previous
            # point, create a new header for this section of the output.
            if curr_point != prev_point:
                point_question = points_dict.get(curr_point)
                combined_output += f"### Point {curr_point} Criterion\n\n{point_question}\n\n"
                prev_point = curr_point
            # Increment the per-point question counter since a new question is processed
            per_point_counters[curr_point] = per_point_counters.get(curr_point, 0) + 1

            # Add the formatted header and information associated with
            # the current question to the combined output.
            display_num = per_point_counters[curr_point]
            heading = f"Point {curr_point} - Question {display_num}"
            combined_output += f"## {heading}\n\n"
            combined_output += f"**{row['question']}**\n\n"
            combined_output += f"### Research Findings\n\n"
            combined_output += f"{results.get(question_index + 1)}\n\n"
            combined_output += f"---\n\n"

        # Evaluate pass/fail per point
        print("\nRunning evaluator (GPT-5 high reasoning) for Pass/Fail assessment...")
        eval_summary_md = evaluate_points(combined_output)
        if eval_summary_md:
            print("Evaluation complete. Summary will be inserted at top of markdown.")
        else:
            print("No point sections detected for evaluation.")

        # Save the combined output and evaluation summary to the
        # final markdown file.
        print(f"\nSaving research to Markdown file...")
        md_filename = save_to_markdown(
            ticker,
            combined_output,
            point_value if point_value is not None else 'all',
            evaluation_summary=eval_summary_md
        )
        print(f"Markdown saved: {md_filename}")
        
        # Print the final summary of research statistics to
        # provide a last update for the user.
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"  Questions researched: {len(filtered_df)}")
        if minutes > 0:
            print(f"  Total time: {minutes}m {seconds:.1f}s")
        else:
            print(f"  Total time: {seconds:.1f}s")
        print(f"{'='*60}\n")


def main(): 
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
