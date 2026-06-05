import sys
from loguru import logger
from src.neo4j_loader import Neo4jLoader

def main():
    logger.info("Initializing Neo4j database reset...")
    loader = Neo4jLoader()
    
    try:
        loader.connect()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j. Reset aborted: {e}")
        sys.exit(1)

    try:
        with loader.driver.session() as session:
            logger.warning("Deleting all nodes and relationships from Neo4j (DETACH DELETE)...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully.")
            
        # Recreate uniqueness constraints and indexes
        loader.create_constraints()
        loader.create_indexes()
        logger.info("Database reset completed successfully.")
    except Exception as e:
        logger.error(f"Error during database reset: {e}")
        sys.exit(1)
    finally:
        loader.close()

if __name__ == "__main__":
    main()
