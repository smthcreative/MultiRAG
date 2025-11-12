import sys
import logging
from img_desc_generation import main as desc_main
from ingest import main as scrape_main
from merging_data import main as merge_main
from indexing import main as index_main
from config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if required environment variables are set."""
    try:
        config.validate()
        logger.info("Configuration validated")
        return True
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set up your .env file with required API keys")
        return False


def run_scraping():
    """Run web scraping."""
    logger.info("STEP 1: Web Scraping")
    try:
        scrape_main()
        logger.info("Scraping completed")
        return True
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return False


def run_image_description():
    """Run image description generation."""
    logger.info("STEP 2: Image Description Generation")
    try:
        desc_main()
        logger.info("Image descriptions generated")
        return True
    except Exception as e:
        logger.error(f"Image description generation failed: {e}")
        return False


def run_merging():
    """Run data merging."""
    logger.info("STEP 3: Data Merging")
    try:
        merge_main()
        logger.info("Data merged successfully")
        return True
    except Exception as e:
        logger.error(f"Data merging failed: {e}")
        return False


def run_indexing():
    """Run vector store indexing."""
    logger.info("STEP 4: Vector Store Indexing")
    try:
        index_main()
        logger.info("Indexing completed")
        return True
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False


def main():
    """Run complete pipeline."""
    logger.info("MultiRAG Pipeline Runner")
    logger.info("This will execute all steps: scraping, descriptions, merging, indexing\n")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Please fix configuration issues before running pipeline")
        sys.exit(1)

    # Run pipeline steps
    steps = [
        ("Scraping", run_scraping),
        ("Image Description", run_image_description),
        ("Merging", run_merging),
        ("Indexing", run_indexing)
    ]

    for step_name, step_func in steps:
        success = step_func()
        if not success:
            logger.error(f"\nPipeline failed at step: {step_name}")
            logger.error("Please fix the errors and re-run the pipeline")
            sys.exit(1)

    # Success
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")



if __name__ == "__main__":
    main()