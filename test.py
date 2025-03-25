import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))