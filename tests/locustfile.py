from locust import HttpUser, task, between
import random

QUESTIONS = [
    "How many artists are there?",
    "What is the total revenue from all invoices?",
    "Which artist has the most albums?",
    "How many tracks are in the Rock genre?",
    "How many customers are from the USA?",
    "What is the average invoice total?",
    "How many invoices are there in total?",
    "How many albums are in the database?",
    "Which country has the most customers?",
    "What is the most expensive track price?",
]

class QueryCraftUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query(self):
        self.client.post("/query", json={"question": random.choice(QUESTIONS)}, timeout=30)

    @task(3)
    def health(self):
        self.client.get("/health")