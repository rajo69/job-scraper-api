# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install Python dependencies
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download the NLTK data during the build process (Corrected Method)
RUN python -m nltk.downloader punkt_tab stopwords

# 6. Copy your application code into the container
COPY . .

# 7. Expose the port the app runs on
EXPOSE 8080

# 8. Define the command to run your app using uvicorn
# The host 0.0.0.0 is required to be accessible from outside the container
# Cloud Run will automatically use the port specified by the PORT env var, or 8080 by default.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
