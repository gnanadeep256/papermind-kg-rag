import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    print(f"Attempting to connect to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print("Success! Successfully connected and authenticated with Neo4j.")
        driver.close()
    except Exception as e:
        print(f"Error: Connection to Neo4j failed. Details: {e}")
        print("Please check that your Neo4j service is running and credentials in .env are correct.")

if __name__ == "__main__":
    main()
