from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextBrowser, QListWidget
)
from PySide6.QtCore import Qt
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import sys

# Load data from Excel file
excel_file = "Service.xlsx"
if not os.path.exists(excel_file):
    raise FileNotFoundError(f"Excel file '{excel_file}' not found.")

# Load organizations from Excel
organizations_df = pd.read_excel(excel_file, sheet_name='Sheet4')
organizations = [str(org).strip() for org in organizations_df['Agency'] if pd.notna(org)]
organization_urls = organizations_df['Agency URL'].tolist()

# Initialize embedding model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')

# Organization embeddings
organization_embeddings = model.encode(organizations, convert_to_tensor=False).astype("float32")
organization_index = faiss.IndexFlatL2(organization_embeddings.shape[1])
organization_index.add(organization_embeddings)

# Titles for the 8 pages
webpage_titles = [
    "*Ongoing* Volunteers Needed for CME LPGA Golf Event",
    "*Ongoing* Bilingual Volunteer Needed for Front Desk Administration at the Salvation Army in Bonita Springs",
    "Volunteers Needed for Processing Donations at Palm Warehouse for the Salvation Army *Ongoing*",
    "Volunteers Needed for Holiday Workshop with Babcock Ranch Lifestyle Department",
    "Volunteers Needed for Breakfast Prep at Center of Hope for the Salvation Army",
    "Volunteers Needed for Thanksgiving 5K with Babcock Ranch Lifestyle Department",
    "Volunteer Decoys & Role Players Needed for the Transportation Security Administration (TSA)",
    "Executive Cabinet"
]

# Placeholder content for webpages
webpage_contents = [
    """
    <p>Date and Time:<br>Sunday, November 24 2024 at 8:00 AM EST to 5:00 PM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>2650 Tiburon Drive Naples, FL 34109</p>
    <p>Description:<br>
    Adrenaline USA Volleyball needs assistance running the concession stand at the golf tournament.
    Two shifts available each day from November 21st-24th: 8 AM - 12 PM and 12 PM - 5 PM.<br>
    *Volunteers are required to wear patriotic attire*</p>
    <p>To sign up, email Coach Hall: <a href="mailto:Adrenalineusavolleyball@gmail.com">Adrenalineusavolleyball@gmail.com</a><br>
    <a href="http://www.adrenalineusavolleyball.com">http://www.adrenalineusavolleyball.com</a></p>
    <p>Students who complete this event gain skills in Teamwork/Collaboration, Professionalism/Work Ethic, Career Management.</p>
    <p>SDGs: Good Health and Well-being (3), Decent Work and Economic Growth (8)</p>
    """,
    """
    <p>Date and Time:<br>Monday, November 25 2024 at 9:00 AM EST to 4:00 PM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>Salvation Army Bonita Springs Service Center</p>
    <p>Description:<br>
    Assist at the front desk of the Salvation Army Bonita Springs Service Center with tasks like answering phones, 
    transferring calls, greeting guests, taking messages, and light office duties. Shifts every Monday and Wednesday:
    9:00 AM - 12:00 PM and 1:00 PM - 4:00 PM.</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving, Oral/Written Communications, and Digital Technology.</p>
    <p>SDGs: No Poverty (1), Zero Hunger (2), and Good Health and Well-being (3)</p>
    """,
    """
    <p>Date and Time:<br>Monday, November 25 2024 at 8:00 AM EST to 12:00 PM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>Warehouse</p>
    <p>Description:<br>
    Sort donated items for the Family Shelter and Community Resource Center. Shifts available Monday-Friday, 
    8:00 AM to 12:00 PM. Comfortable, conservative, and casual dress code required.</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving, Teamwork/Collaboration, and Professionalism/Work Ethic.</p>
    <p>SDGs: No Poverty (1), Zero Hunger (2), and Good Health and Well-being (3)</p>
    """,
    """
    <p>Date and Time:<br>Tuesday, November 26 2024 at 5:00 PM EST to 8:00 PM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>42911 Lake Babcock Dr, Babcock Ranch, FL 33982</p>
    <p>Description:<br>
    Assist with arts and crafts, decorating, and creating a warm environment during a festive holiday workshop.
    No experience needed; bring your holiday spirit and help spread cheer!</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving, Oral/Written Communications, and Teamwork/Collaboration.</p>
    <p>SDGs: Good Health and Well-being (3), Affordable and Clean Energy (7), and Life on Land (15)</p>
    """,
    """
    <p>Date and Time:<br>Monday, November 25 2024 at 5:00 AM EST to Friday, November 29 2024 at 8:00 AM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>The Salvation Army Center of Hope</p>
    <p>Description:<br>
    Assist with meal preparation, kitchen cleaning, and stocking supplies. Closed-toed shoes required for safety.</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving, Oral/Written Communications, and Teamwork/Collaboration.</p>
    <p>SDGs: No Poverty (1), Zero Hunger (2), and Good Health and Well-being (3)</p>
    """,
    """
    <p>Date and Time:<br>Saturday, November 30 2024 at 7:00 AM EST to 10:00 AM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>42911 Lake Babcock Drive Babcock Ranch, FL 33982</p>
    <p>Description:<br>
    Help with the Thanksgiving 5K in a sustainable community! Tasks include cheering, water station management, 
    and participant assistance. Bring your energy and holiday spirit!</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving, Oral/Written Communications, and Teamwork/Collaboration.</p>
    <p>SDGs: Good Health and Well-being (3), Affordable and Clean Energy (7), and Life on Land (15)</p>
    """,
    """
    <p>Date and Time:<br>Thursday, November 28 2024 at 9:00 AM EST to Saturday, November 30 2024 at 1:00 PM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>Southwest Florida International Airport (RSW)</p>
    <p>Description:<br>
    Assist TSA K9 teams by acting as decoys or role players. Earn service hours while supporting national security efforts.</p>
    <p>Students who complete this event gain skills in Critical Thinking/Problem Solving and Teamwork/Collaboration.</p>
    <p>SDGs: Sustainable Cities and Communities (11), Peace, Justice, and Strong Institutions (16)</p>
    """,
    """
    <p>Date and Time:<br>Friday, November 22 2024 at 7:00 AM EST to Monday, November 25 2024 at 8:00 AM EST</p>
    <p>Add To Google Calendar | iCal/Outlook</p>
    <p>Location:<br>Cohen 249</p>
    <p>Description:<br>Weekly Executive Branch Meeting.</p>
    """
]

# Webpage content embeddings
content_embeddings = model.encode(webpage_contents, convert_to_tensor=False).astype("float32")
content_index = faiss.IndexFlatL2(content_embeddings.shape[1])
content_index.add(content_embeddings)

# Matching functions
def match_content_to_organization(content, top_k=1, similarity_threshold=1):
    content_embedding = model.encode([content], convert_to_tensor=False).astype("float32")
    distances, indices = organization_index.search(content_embedding, top_k)

    return [
        (organizations[i], organization_urls[i], distances[0][j])
        for j, i in enumerate(indices[0]) if distances[0][j] <= similarity_threshold
    ]

def match_query_to_events(query, top_k=5, similarity_threshold=1.5):
    query_embedding = model.encode([query], convert_to_tensor=False).astype("float32")
    distances, indices = content_index.search(query_embedding, top_k)

    # Filter results based on similarity threshold
    return [
        (webpage_titles[i], webpage_contents[i], distances[0][j])
        for j, i in enumerate(indices[0]) if distances[0][j] <= similarity_threshold
    ]

# PySide6 GUI
class OrganizationMatcher(QWidget):
    def __init__(self):
        super().__init__()

        # Set up layout
        self.setWindowTitle("Organization and Event Matcher")
        self.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout()

        # Search bar and suggestions
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search for an organization or event...")
        self.search_bar.textChanged.connect(self.update_suggestions)
        layout.addWidget(self.search_bar)

        self.suggestion_list = QListWidget()
        self.suggestion_list.setMaximumHeight(100)
        self.suggestion_list.itemClicked.connect(self.fill_search_bar)
        layout.addWidget(self.suggestion_list)

        # Webpage buttons
        for i, title in enumerate(webpage_titles):
            button = QPushButton(title)
            button.clicked.connect(lambda _, idx=i: self.open_mock_page(idx))
            layout.addWidget(button)

        # Content display area
        self.content_browser = QTextBrowser()
        layout.addWidget(self.content_browser)

        self.setLayout(layout)

    def update_suggestions(self):
        query = self.search_bar.text().strip()
        if not query:
            self.suggestion_list.clear()
            self.content_browser.setHtml("<p>Enter a query to see suggestions and matches.</p>")
            return

        # Find matching organizations
        org_matches = match_content_to_organization(query, top_k=5)
        self.suggestion_list.clear()
        for org, _, _ in org_matches:
            self.suggestion_list.addItem(org)

        # Find matching events with a threshold
        event_matches = match_query_to_events(query, top_k=3, similarity_threshold=1.2)
        content = "<h2>Most Similar Events:</h2><ul>"
        if event_matches:
            for title, _, dist in event_matches:
                content += f"<li>{title} - Similarity Score: {dist:.4f}</li>"
        else:
            content += "<li>No similar events found within the threshold.</li>"
        content += "</ul>"

        self.content_browser.setHtml(content)

    def fill_search_bar(self, item):
        self.search_bar.setText(item.text())

    def open_mock_page(self, index):
        # Get the base content
        content = f"<h1>{webpage_titles[index]}</h1><p>{webpage_contents[index]}</p>"

        # Match content to organizations
        matches = match_content_to_organization(webpage_contents[index])
        if matches:
            content += "<h2>Detected Organizations:</h2><ul>"
            for org, url, dist in matches:
                content += f'<li><b>{org}</b> - Similarity Score: {dist:.4f}<br>Full URL: <a href="{url}">{url}</a></li>'
            content += "</ul>"
        else:
            content += "<p>No matching organizations found for this content.</p>"

        self.content_browser.setHtml(content)

    def update_suggestions(self):
        query = self.search_bar.text().strip()
        if not query:
            self.suggestion_list.clear()
            return

        # Find matching organizations
        matches = match_content_to_organization(query, top_k=5)
        self.suggestion_list.clear()
        for org, _, _ in matches:
            self.suggestion_list.addItem(org)

        # Find matching events
        events = match_query_to_events(query, top_k=3)
        content = "<h2>Most Similar Events:</h2><ul>"
        for title, _, dist in events:
            content += f"<li>{title} - Similarity Score: {dist:.4f}</li>"
        content += "</ul>"

        self.content_browser.setHtml(content)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OrganizationMatcher()
    window.show()
    app.exec()
