# Use an official lightweight Python image as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code and data into the container.
COPY . .

# Expose the port that the app will run on.
EXPOSE 8080

# The command to run when the container starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]