# Use a standard, slim Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (app.py) into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 8080

# Command to run the application using Gunicorn (a production-ready server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]