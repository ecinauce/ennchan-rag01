# ennchan_interface_cmd/app.py
import time
import os
import sys
import logging
import argparse
import io
import contextlib
from ennchan_rag.ask import ask

class QuietFilter(logging.Filter):
    def filter(self, record):
        return False

@contextlib.contextmanager
def suppress_output(verbose=False):
    """Context manager to suppress both stdout and stderr output."""
    if verbose:
        yield  # Don't suppress anything in verbose mode
        return
        
    # Save original stdout/stderr and logging configuration
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_filters = root_logger.filters.copy()
    original_handlers = root_logger.handlers.copy()
    
    try:
        # Redirect stdout/stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Silence root logger
        root_logger.setLevel(logging.CRITICAL)
        quiet_filter = QuietFilter()
        root_logger.addFilter(quiet_filter)
        
        # Silence all other loggers
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        yield
    finally:
        # Restore stdout/stderr and logging configuration
        sys.stdout = save_stdout
        sys.stderr = save_stderr
        root_logger.setLevel(original_level)
        root_logger.filters = original_filters
        root_logger.handlers = original_handlers

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def clean_output(output):
    return output.split("Answer:")[-1].strip()

def print_header():
    """Print the application header."""
    print("=" * 80)
    print("EnnchanRAG Command Line Interface".center(80))
    print("Type 'exit', 'quit', 'close', or 'q' to exit".center(80))
    print("=" * 80)
    print()

def main():    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EnnchanRAG Command Line Interface")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        # Configure verbose logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print("Verbose mode enabled. Debug information will be displayed.")
    else:
        # Configure minimal logging
        logging.basicConfig(level=logging.CRITICAL)
 
    clear_screen()
    print_header()
    
    run = True
    while run:
        try:
            prompt = input("\033[1mUser:\033[0m ")
            if prompt.lower() in ["exit", "quit", "close", "q"]:
                print("\nThank you for using EnnchanRAG. Goodbye!")
                run = False
            elif prompt.strip() == "":
                continue
            else:
                start_time = time.time()
                print("\033[90mThinking...\033[0m")
                
                # Suppress all output if not in verbose mode
                with suppress_output(args.verbose):
                    reply = ask(prompt, "..\\config.json")
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Clear the "Thinking..." line
                sys.stdout.write("\033[F\033[K")
                
                print(f"\033[1;34mAssistant:\033[0m {clean_output(reply)}")
                print(f"\033[90m(Response time: {runtime:.2f} seconds)\033[0m\n")
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Exiting...")
            run = False
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")
            if args.verbose:
                import traceback
                print("\033[31m" + traceback.format_exc() + "\033[0m")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
