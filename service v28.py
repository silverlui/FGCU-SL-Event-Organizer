from bs4 import BeautifulSoup
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextBrowser, QListWidget
)
from PySide6.QtCore import QTimer, QThread, Signal
from sentence_transformers import SentenceTransformer
import requests
from xml.etree import ElementTree as ET
import pandas as pd
import faiss
import sys
import os
from datetime import datetime


class RSSFetcher(QThread):
    data_fetched = Signal(list)

    def run(self):
        url = "https://getinvolved.fgcu.edu/events.rss"
        try:
            response = requests.get(url)
            response.raise_for_status()
            events = self.parse_rss(response.content)
            self.data_fetched.emit(events)
        except requests.RequestException:
            self.data_fetched.emit([])

    @staticmethod
    def parse_rss(content):
        root = ET.fromstring(content)
        items = []
        namespace = {'events': 'events'}

        for item in root.findall(".//item"):
            start_date = item.findtext("events:start", namespaces=namespace)
            try:
                parsed_start = datetime.strptime(start_date, "%a, %d %b %Y %H:%M:%S %Z") if start_date else None
            except ValueError:
                parsed_start = None

            event = {
                "title": item.findtext("title"),
                "description": item.findtext("description"),
                "link": item.findtext("link"),
                "start": start_date,
                "end": item.findtext("events:end", namespaces=namespace),
                "location": item.findtext("events:location", namespaces=namespace),
                "parsed_start": parsed_start,
            }
            items.append(event)
        return items


class EventBrowser(QWidget):
    def __init__(self, excel_file):
        super().__init__()

        self.setWindowTitle("FGCU Events Browser")
        self.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout()

        # Search bar and suggestion list
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search for organizations...")
        self.search_bar.textChanged.connect(self.update_suggestions)
        layout.addWidget(self.search_bar)

        self.suggestion_list = QListWidget()
        self.suggestion_list.setMaximumHeight(100)
        self.suggestion_list.itemClicked.connect(self.handle_agency_selection)
        layout.addWidget(self.suggestion_list)

        # Agency details and event filtering
        self.agency_details = QTextBrowser()
        self.agency_details.setPlaceholderText("Select an agency to view details.")
        layout.addWidget(self.agency_details)

        # Button to reset event list
        self.list_all_button = QPushButton("List All Events")
        self.list_all_button.clicked.connect(self.reset_event_list)
        layout.addWidget(self.list_all_button)

        # Event list and details
        self.event_list = QListWidget()
        self.event_list.itemClicked.connect(self.display_event_details)
        layout.addWidget(self.event_list)

        self.event_details = QTextBrowser()
        layout.addWidget(self.event_details)

        self.setLayout(layout)

        # Initialize attributes
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.events = []
        self.filtered_events = []
        self.event_index = None

        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file '{excel_file}' not found.")
        self.excel_data = pd.read_excel(excel_file, sheet_name='Sheet4')
        self.suggestion_embeddings = self.model.encode(
            self.excel_data['Agency'], convert_to_tensor=False).astype("float32")
        self.suggestion_index = faiss.IndexFlatL2(self.suggestion_embeddings.shape[1])
        self.suggestion_index.add(self.suggestion_embeddings)

        self.agencies = []

        self.rss_fetcher = RSSFetcher()
        self.rss_fetcher.data_fetched.connect(self.process_events)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.load_data)
        self.refresh_timer.start(60 * 60 * 1000)

        self.load_data()

    def scrape_agencies(self):
        try:
            url = "https://servicelearning.fgcu.edu/CommunityPartnerDatabase/AgenciesList"
            response = requests.get(url)
            response.raise_for_status()
            self.agencies = self.extract_agency_data(response.text)
        except requests.RequestException as e:
            print(f"Failed to scrape agencies: {e}")
            self.agencies = []

    def extract_agency_data(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        agencies = []

        rows = soup.select("table#agencieslist tbody tr")
        for row in rows:
            agency = {
                "Name": row.select_one("td a.edit").text if row.select_one("td a.edit") else "N/A",
                "County": row.select_one("td:nth-child(2)").text.strip() if row.select_one("td:nth-child(2)") else "N/A",
                "Address": row.select_one("td.d-none:nth-child(6)").text.strip() if row.select_one("td.d-none:nth-child(6)") else "N/A",
                "City": row.select_one("td.d-none:nth-child(8)").text.strip() if row.select_one("td.d-none:nth-child(8)") else "N/A",
                "State": row.select_one("td.d-none:nth-child(9)").text.strip() if row.select_one("td.d-none:nth-child(9)") else "N/A",
                "Zip": row.select_one("td.d-none:nth-child(10)").text.strip() if row.select_one("td.d-none:nth-child(10)") else "N/A",
                "Email": row.select_one("td.d-none:nth-child(19)").text.strip() if row.select_one("td.d-none:nth-child(19)") else "N/A",
                "Website": row.select_one("td.d-none:nth-child(20)").text.strip() if row.select_one("td.d-none:nth-child(20)") else "N/A",
            }
            agencies.append(agency)

        return agencies

    def load_data(self):
        self.scrape_agencies()
        self.load_events()

    def load_events(self):
        self.rss_fetcher.start()

    def process_events(self, events):
        if not events:
            self.event_list.addItem("No events found.")
            return

        self.events = sorted(events, key=lambda x: x["parsed_start"] or datetime.max)
        self.filtered_events = list(self.events)
        self.build_faiss_index_for_events()
        self.display_events()

    def build_faiss_index_for_events(self):
        if self.events:
            texts = [f"{event['title']} {event['description']}" for event in self.events]
            embeddings = self.model.encode(texts, convert_to_tensor=False).astype("float32")
            self.event_index = faiss.IndexFlatL2(embeddings.shape[1])
            self.event_index.add(embeddings)

    def display_events(self):
        self.event_list.clear()
        for event in self.filtered_events:
            self.event_list.addItem(event["title"])

    def display_event_details(self, item):
        selected_title = item.text()
        event = next((e for e in self.events if e["title"] == selected_title), None)
        if event:
            matching_agencies = [
                agency for agency in self.agencies
                if agency["Name"].lower() in event['description'].lower()
            ]

            if matching_agencies:
                agency_details = "\n".join([
                    f"<b>{key}:</b> {value}" for agency in matching_agencies
                    for key, value in agency.items()
                ])
                agency_html = f"<h3>Associated Agencies:</h3>{agency_details}"
            else:
                agency_html = "<h3>No associated agencies found.</h3>"

            details = f"""
                <h2>{event['title']}</h2>
                <p><b>Start:</b> {event['start']}</p>
                <p><b>End:</b> {event['end']}</p>
                <p><b>Location:</b> {event['location']}</p>
                <p><b>Description:</b> {event['description']}</p>
                <p><a href="{event['link']}">Event Link</a></p>
                {agency_html}
            """
            self.event_details.setHtml(details)
        else:
            self.event_details.setText("Event details not found.")

    def handle_agency_selection(self, item):
        agency_name = item.text().split(" (Score:")[0]

        # Display agency details
        agency = next((a for a in self.agencies if a["Name"] == agency_name), None)
        if agency:
            details = "\n".join([f"<b>{key}:</b> {value}" for key, value in agency.items()])
            self.agency_details.setHtml(details)
        else:
            self.agency_details.setText("Agency details not found.")

        # Filter events
        self.filtered_events = [
            event for event in self.events if agency_name.lower() in event['description'].lower()
        ]
        self.display_events()

    def update_suggestions(self):
        query = self.search_bar.text().strip()
        if query:
            query_embedding = self.model.encode([query], convert_to_tensor=False).astype("float32")
            distances, indices = self.suggestion_index.search(query_embedding, k=10)
            self.suggestion_list.clear()
            for idx, dist in zip(indices[0], distances[0]):
                if dist < float('inf'):
                    agency = self.excel_data.iloc[idx]["Agency"]
                    self.suggestion_list.addItem(f"{agency} (Score: {dist:.2f})")
        else:
            self.suggestion_list.clear()

    def reset_event_list(self):
        self.filtered_events = list(self.events)
        self.display_events()


if __name__ == "__main__":
    excel_file = "Service.xlsx"
    app = QApplication(sys.argv)
    window = EventBrowser(excel_file)
    window.show()
    app.exec()
